#!/usr/bin/python

# Copyright 2017 Google Inc.
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
r"""Filter and revise a collection of TensorFlow Example protos.

This pipeline is useful when a full dataset has been preprocessed but for
a subsequent experiment you wish to use only a subset of the examples and/or
modify the labels.  The subset to be copied is determined by the list of
sample names returned by the metadata query.

USAGE:
  python -m trainer.revise_preprocessed_data \
    --setup_file ./setup.py \
    --project ${PROJECT_ID} \
    --input ${BUCKET}/sgdp \
    --metadata preprocess/sgdp_metadata_remap_labels.jinja \
    --output ${BUCKET}/sgdp_relabeled_subset
"""

import datetime
import logging
import os

import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import WorkerOptions
from jinja2 import Template
import tensorflow as tf

import trainer.ancestry_metadata_encoder as metadata_encoder
import trainer.feature_encoder as encoder
import trainer.util as util

METADATA_QUERY_REPLACEMENTS = {
    'KEY_COLUMN': encoder.KEY_COLUMN,
    'POPULATION_COLUMN': metadata_encoder.POPULATION_COLUMN,
    'SUPER_POPULATION_COLUMN': metadata_encoder.SUPER_POPULATION_COLUMN,
    'GENDER_COLUMN': metadata_encoder.GENDER_COLUMN,
}


def filter_and_revise_example(serialized_example, samples_metadata):
  """Filter and revise a collection of existing TensorFlow examples.

  Args:
    serialized_example: the example to be revised and/or filtered
    samples_metadata: dictionary of metadata for all samples

  Returns:
    A list containing the revised example or the empty list if the
    example should be removed from the collection.
  """
  example = tf.train.Example.FromString(serialized_example)
  sample_name = example.features.feature[
      encoder.SAMPLE_NAME_FEATURE].bytes_list.value[0]
  logging.info('Checking ' + sample_name)
  if sample_name not in samples_metadata:
    logging.info('Omitting ' + sample_name)
    return []

  revised_features = {}
  # Initialize with current example features.
  revised_features.update(example.features.feature)
  # Overwrite metadata features.
  revised_features.update(
      metadata_encoder.metadata_to_ancestry_features(
          samples_metadata[sample_name]))
  return [
      tf.train.Example(features=tf.train.Features(feature=revised_features))
  ]


class ReviseOptions(PipelineOptions):

  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument(
        '--input',
        dest='input',
        required=True,
        help='Input directory holding the previously preprocessed examples.')
    parser.add_argument(
        '--metadata',
        dest='metadata',
        required=True,
        help='Jinja file holding the metadata query.')
    parser.add_argument(
        '--output',
        dest='output',
        required=True,
        help='Output directory to which to write filtered and revised '
        'examples.')


def run(argv=None):
  """Runs the revise preprocessed data pipeline.

  Args:
    argv: Pipeline options as a list of arguments.
  """
  pipeline_options = PipelineOptions(flags=argv)
  revise_options = pipeline_options.view_as(ReviseOptions)
  cloud_options = pipeline_options.view_as(GoogleCloudOptions)
  output_dir = os.path.join(revise_options.output,
                            datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  pipeline_options.view_as(SetupOptions).save_main_session = True
  pipeline_options.view_as(
      WorkerOptions).autoscaling_algorithm = 'THROUGHPUT_BASED'
  cloud_options.staging_location = os.path.join(output_dir, 'tmp', 'staging')
  cloud_options.temp_location = os.path.join(output_dir, 'tmp')
  cloud_options.job_name = 'relabel-examples-%s' % (
      datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

  metadata_query = str(
      Template(open(revise_options.metadata, 'r').read()).render(
          METADATA_QUERY_REPLACEMENTS))
  logging.info('metadata query : %s', metadata_query)

  with beam.Pipeline(options=pipeline_options) as p:
    # Gather our sample metadata into a python dictionary.
    samples_metadata = (
        p
        | 'ReadSampleMetadata' >> beam.io.Read(
            beam.io.BigQuerySource(query=metadata_query, use_standard_sql=True))
        | 'TableToDictionary' >> beam.CombineGlobally(
            util.TableToDictCombineFn(key_column=encoder.KEY_COLUMN)))

    # Read the tf.Example protos into a PCollection.
    examples = p | 'ReadExamples' >> tfrecordio.ReadFromTFRecord(
        file_pattern=revise_options.input,
        compression_type=CompressionTypes.GZIP)

    # Filter the TensorFlow Example Protocol Buffers.
    filtered_examples = (examples | 'ReviseExamples' >> beam.FlatMap(
        lambda example, samples_metadata:
        filter_and_revise_example(example, samples_metadata),
        beam.pvalue.AsSingleton(samples_metadata)))

    # Write the subset of tf.Example protos to Cloud Storage.
    _ = (filtered_examples
         | 'SerializeExamples' >>
         beam.Map(lambda example: example.SerializeToString())
         | 'WriteExamples' >> tfrecordio.WriteToTFRecord(
             file_path_prefix=os.path.join(output_dir, 'examples'),
             compression_type=CompressionTypes.GZIP,
             file_name_suffix='.tfrecord.gz'))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
