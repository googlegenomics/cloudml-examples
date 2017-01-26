#!/usr/bin/python

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Pipeline to convert variant data from BigQuery to TensorFlow Example protos.

USAGE:
  python preprocess_data.py \
    --requirements_file python_dependencies.txt \
    --output gs://MY-BUCKET/variant-preprocesss \
    --project MY-PROJECT \
    --metadata 1000_genomes_metadata.jinja \
    --data 1000_genomes_phase3_b37_limit10.jinja
"""

import argparse
import datetime
import logging
import os

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.pvalue import AsSingleton
from apache_beam.utils.options import GoogleCloudOptions
from apache_beam.utils.options import PipelineOptions
from apache_beam.utils.options import SetupOptions
from apache_beam.utils.options import WorkerOptions
from jinja2 import Template
import tensorflow as tf
from trainer.feature_encoder import BinnedFeatureEncoder
from trainer.feature_encoder import FeatureEncoder
import trainer.util as util

import google.cloud.ml as ml
import google.cloud.ml.dataflow.io.tfrecordio as tfrecordio
from google.cloud.ml.io.coders import ExampleProtoCoder


# Jinja template replacements to decouple column names from the source
# tables from the dictionart keys used in this pipeline.

METADATA_QUERY_REPLACEMENTS = {
    'KEY_COLUMN': FeatureEncoder.KEY_COLUMN,
    'POPULATION_COLUMN': FeatureEncoder.POPULATION_COLUMN,
    'SUPER_POPULATION_COLUMN': FeatureEncoder.SUPER_POPULATION_COLUMN,
    'GENDER_COLUMN': FeatureEncoder.GENDER_COLUMN,
}

DATA_QUERY_REPLACEMENTS = {
    'KEY_COLUMN': FeatureEncoder.KEY_COLUMN,
    'CONTIG_COLUMN': FeatureEncoder.CONTIG_COLUMN,
    'START_COLUMN': FeatureEncoder.START_COLUMN,
    'END_COLUMN': FeatureEncoder.END_COLUMN,
    'REF_COLUMN': FeatureEncoder.REF_COLUMN,
    'ALT_COLUMN': FeatureEncoder.ALT_COLUMN,
    'ALT_NUM_COLUMN': FeatureEncoder.ALT_NUM_COLUMN,
    'FIRST_ALLELE_COLUMN': FeatureEncoder.FIRST_ALLELE_COLUMN,
    'SECOND_ALLELE_COLUMN': FeatureEncoder.SECOND_ALLELE_COLUMN
}


def variants_to_examples(input_data, samples_metadata, feature_encoder):
  """Converts variants to TensorFlow Example protos.

  Args:
    input_data: variant call dictionary objects with keys from
      DATA_QUERY_REPLACEMENTS
    samples_metadata: metadata dictionary objects with keys from
      METADATA_QUERY_REPLACEMENTS
    feature_encoder: the feature encoder instance to use to convert the source
      data into TensorFlow Example protos.

  Returns:
    TensorFlow Example protos.
  """
  variant_kvs = input_data | 'bucketVariants' >> beam.Map(
      lambda row: (row[FeatureEncoder.KEY_COLUMN], row))

  sample_variant_kvs = variant_kvs | 'groupBySample' >> beam.GroupByKey()

  examples = (
      sample_variant_kvs
      | 'samplesToExamples' >> beam.Map(
          lambda (key, vals), samples_metadata: feature_encoder.sample_variants_to_example(key, vals, samples_metadata),
          AsSingleton(samples_metadata)))

  return examples


def run(argv=None):
  """Runs the variant preprocess pipeline.

  Args:
    argv: Pipeline options as a list of arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output',
      required=True,
      help='Output directory to which to write results.')
  parser.add_argument(
      '--data',
      required=True,
      help='Jinja file holding the query for the sample data.')
  parser.add_argument(
      '--metadata',
      required=True,
      help='Jinja file holding the query for the sample metadata.')
  parser.add_argument(
      '--hethom_words',
      dest='add_hethom',
      action='store_true',
      help='Add variant heterozygous/homozygous "word".'
      'Defaults to true.')
  parser.add_argument(
      '--no_hethom_words',
      dest='add_hethom',
      action='store_false',
      help='Do not add variant heterozygous/homozygous "word".'
      'Defaults to true.')
  parser.set_defaults(add_hethom=True)
  parser.add_argument('--bin_size', type=int)
  known_args, pipeline_args = parser.parse_known_args(argv)
  output_dir = os.path.join(known_args.output,
                            datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True
  pipeline_options.view_as(SetupOptions).extra_packages = [ml.sdk_location]
  pipeline_options.view_as(WorkerOptions).autoscaling_algorithm = 'THROUGHPUT_BASED'
  pipeline_options.view_as(GoogleCloudOptions).staging_location = os.path.join(
      output_dir, 'tmp', 'staging')
  pipeline_options.view_as(GoogleCloudOptions).temp_location = os.path.join(
      output_dir, 'tmp')
  pipeline_options.view_as(
      GoogleCloudOptions
  ).job_name = 'preprocess-varianteatures-%s' % (datetime.datetime.now(
  ).strftime('%y%m%d-%H%M%S'))

  metadata_query = str(
      Template(open(known_args.metadata, 'r').read()).render(
          METADATA_QUERY_REPLACEMENTS))
  logging.info('metadata query : %s', metadata_query)

  data_query = str(
      Template(open(known_args.data, 'r').read()).render(
          DATA_QUERY_REPLACEMENTS))
  logging.info('data query : %s', data_query)

  beam.coders.registry.register_coder(tf.train.Example, ExampleProtoCoder)
  p = beam.Pipeline(options=pipeline_options)

  # Gather our sample metadata into a python dictionary.
  samples_metadata = (
      p
      | beam.io.Read(
          'readSampleMetadata',
          beam.io.BigQuerySource(
              query=metadata_query, use_standard_sql=True))
      | 'tableToDictionary' >> beam.CombineGlobally(
          util.TableToDictCombineFn(key_column=FeatureEncoder.KEY_COLUMN)))

  # Read the table rows into a PCollection.
  rows = p | beam.io.Read(
      'readVariants', beam.io.BigQuerySource(query=data_query,
                                             use_standard_sql=True))

  feature_encoder = FeatureEncoder(add_hethom=known_args.add_hethom)
  if known_args.bin_size is not None:
    feature_encoder = BinnedFeatureEncoder(
        add_hethom=known_args.add_hethom, bin_size=known_args.bin_size)

  # Convert the data into TensorFlow Example Protocol Buffers.
  examples = variants_to_examples(
      rows, samples_metadata, feature_encoder=feature_encoder)

  # Write the serialized compressed protocol buffers to Cloud Storage.
  _ = examples | beam.io.Write(
      'writeExamples',
      tfrecordio.WriteToTFRecord(
          file_path_prefix=os.path.join(output_dir, 'examples'),
          compression_type=fileio.CompressionTypes.GZIP,
          coder=ExampleProtoCoder(),
          file_name_suffix='.tfrecord.gz'))

  # Actually run the pipeline (all operations above are deferred).
  p.run()


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
