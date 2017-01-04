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

"""Pipeline to convert variant data from BigQuery to TensorFlow Example protos.

USAGE:
  python preprocess_data.py \
    --requirements_file python_dependencies.txt \
    --output gs://MY-BUCKET/variant-preprocesss \
    --project MY-PROJECT \
    --metadata 1000_genomes_metadata.jinja \
    --data 1000_genomes_phase3_b37_limit10.jinja
"""

import argparse
import collections
from collections import defaultdict
import datetime
import logging
import os
import struct

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.pvalue import AsSingleton
from apache_beam.transforms import core
from apache_beam.utils.options import GoogleCloudOptions
from apache_beam.utils.options import PipelineOptions
from apache_beam.utils.options import SetupOptions
from apache_beam.utils.options import WorkerOptions
import farmhash
from jinja2 import Template
import tensorflow as tf

import google.cloud.ml as ml
import google.cloud.ml.dataflow.io.tfrecordio as tfrecordio
from google.cloud.ml.io.coders import ExampleProtoCoder

# Values to use for missing data.
NA_STRING = 'NA'
NA_INTEGER = -1

# Decouple column names from the source tables from the
# keys used in this pipeline by using jinja templates
# for the queries and a replacement mapping.
KEY_COLUMN = 'key'
POPULATION_COLUMN = 'pop'
SUPER_POPULATION_COLUMN = 'sup'
GENDER_COLUMN = 'sex'
CONTIG_COLUMN = 'contig'
START_COLUMN = 'start_pos'
END_COLUMN = 'end_pos'
REF_COLUMN = 'ref'
ALT_COLUMN = 'alt'
ALT_NUM_COLUMN = 'alt_num'
FIRST_ALLELE_COLUMN = 'first'
SECOND_ALLELE_COLUMN = 'second'

QUERY_REPLACEMENTS = {
    'KEY_COLUMN': KEY_COLUMN,
    'POPULATION_COLUMN': POPULATION_COLUMN,
    'SUPER_POPULATION_COLUMN': SUPER_POPULATION_COLUMN,
    'GENDER_COLUMN': GENDER_COLUMN,
    'CONTIG_COLUMN': CONTIG_COLUMN,
    'START_COLUMN': START_COLUMN,
    'END_COLUMN': END_COLUMN,
    'REF_COLUMN': REF_COLUMN,
    'ALT_COLUMN': ALT_COLUMN,
    'ALT_NUM_COLUMN': ALT_NUM_COLUMN,
    'FIRST_ALLELE_COLUMN': FIRST_ALLELE_COLUMN,
    'SECOND_ALLELE_COLUMN': SECOND_ALLELE_COLUMN
}

# Population and Super population labels used for training and evaluation
# will always be the ones from 1000 Genomes. If we wish to use another
# dataset for training and evaluation, we'll need to provide the mapping
# from 1000 Genomes labels to those used for the dataset.
SUPER_POPULATIONS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
SUPER_POPULATION_MAP = defaultdict(lambda: NA_INTEGER)
for pop in range(len(SUPER_POPULATIONS)):
  SUPER_POPULATION_MAP[SUPER_POPULATIONS[pop]] = pop

POPULATIONS = [
    'ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN', 'GBR',
    'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL',
    'PUR', 'STU', 'TSI', 'YRI'
]
POPULATION_MAP = defaultdict(lambda: NA_INTEGER)
for pop in range(len(POPULATIONS)):
  POPULATION_MAP[POPULATIONS[pop]] = pop

# Normalize over all possible sex/gender values.  This list is
# not exhaustive and more can be added.
GENDER_MAP = defaultdict(lambda: NA_INTEGER)
GENDER_MAP.update({
    'male': 0,
    'female': 1,
    'Male': 0,
    'Female': 1,
    'm': 0,
    'f': 1,
    'M': 0,
    'F': 1
})


class DefaultToKeyDict(dict):
  """Custom dictionary to use the key as the value for any missing entries."""

  def __missing__(self, key):
    return str(key)

# Normalize reference names, defaulting to the vertabim reference name
# if value is not present in the dictionary.
CONTIG_MAP = DefaultToKeyDict()
CONTIG_MAP.update({('chr' + str(x)): str(x) for x in range(1, 23)})
CONTIG_MAP['chrX'] = 'X'
CONTIG_MAP['chrY'] = 'Y'
CONTIG_MAP['chrM'] = 'MT'
CONTIG_MAP['chrMT'] = 'MT'
CONTIG_MAP['M'] = 'MT'


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class TableToDictCombineFn(core.CombineFn):
  """Create a python dictionary from a BigQuery table.

  This CombineFn reshapes rows from a BigQuery table using the specified key
  column to a Python dictionary.
  """

  def __init__(self, key_column):
    self.key_column = key_column

  def create_accumulator(self):
    return dict()

  def add_input(self, accumulator, element):
    accumulator[element[self.key_column]] = element
    return accumulator

  def add_inputs(self, accumulator, elements):
    for element in elements:
      self.add_input(accumulator, element)
    return accumulator

  def merge_accumulators(self, accumulators):
    final_accumulator = {}
    for accumulator in accumulators:
      final_accumulator.update(accumulator)
    return final_accumulator

  def extract_output(self, accumulator):
    return accumulator


def variant_to_words(variant, add_hethom):
  """Encode a variant as one or more "words".

  Given a variant record with certain expected fields, create two
  "words" that uniquely describe it. The first word would match
  whether the sample is either heterozygous or homozygous for the
  variant.  The second word is more specific and will match one of
  heterozygous or homozygous.

  Args:
    variant: a variant
    add_hethom: whether to add a "word" for the zygosity of the variant

  Returns:
    One or more "words" that represent the variant.
  """
  # Only add words only if the sample has a variant at this site. Since the
  # input data was FLATTENED on alternate_bases, we do this by checking
  # whether either allele value corresponds to the alternate currently
  # under consideration. Note that the values of the first allele and
  # the second allele are genotypes --> which are essentially an index
  # into the alternate_bases repeated field.
  # See http://vcftools.sourceforge.net/VCF-poster.pdf for more detail.
  alt_num = int(variant[ALT_NUM_COLUMN])
  if ((variant[FIRST_ALLELE_COLUMN] != alt_num) and (
      (SECOND_ALLELE_COLUMN not in variant) or
      (variant[SECOND_ALLELE_COLUMN] != alt_num))):
    return []

  # Normalize reference names.
  contig = CONTIG_MAP[variant[CONTIG_COLUMN]]

  variant_word = '_'.join([
      contig, str(variant[START_COLUMN]), str(variant[END_COLUMN]),
      str(variant[REF_COLUMN]), str(variant[ALT_COLUMN])
  ])

  if not add_hethom:
    return [variant_word]

  # Add two words, one for the variant itself and another more specific word
  # (a synonym) regarding heterozygosity/homozygosity of the observation.
  if ((SECOND_ALLELE_COLUMN not in variant) or
      (variant[FIRST_ALLELE_COLUMN] != variant[SECOND_ALLELE_COLUMN])):
    return [variant_word, '_'.join([variant_word, 'het'])]

  return [variant_word, '_'.join([variant_word, 'hom'])]


def sample_variants_to_example(sample, sample_variants, samples_metadata,
                               add_hethom):
  """Convert the variants to TensorFlow Example protocol buffers.

  See also
  https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html

  Args:
    sample: the identifier for the sample
    sample_variants: the sample's variants
    samples_metadata: dictionary of metadata for all samples
    add_hethom: whether to add a "word" for the zygosity of each variant

  Returns:
    A filled in TensorFlow Example proto.
  """
  # Some samples may have no metadata, but we may still want to preprocess
  # the data for prediction use cases.
  metadata = defaultdict(lambda: NA_STRING)
  if sample in samples_metadata:
    metadata.update(samples_metadata[sample])

  variants_by_contig = collections.defaultdict(list)
  for variant in sample_variants:
    # Normalize reference names.
    contig = CONTIG_MAP[variant[CONTIG_COLUMN]]
    words = variant_to_words(variant, add_hethom)
    variants_by_contig[contig].extend(
        # fingerprint64 returns an unsigned int64 but int64 features are
        # signed.  Convert from from unsigned to signed.
        [
            struct.unpack('q', struct.pack('Q', farmhash.fingerprint64(w)))[0]
            for w in words
        ])

  features = {
      'sample_name':
          _bytes_feature(str(sample)),
      # Nomalize population to integer or NA_INTEGER if no match.
      'population':
          _int64_feature(POPULATION_MAP[str(metadata[POPULATION_COLUMN])]),
      # Use verbatim value of population.
      'population_string':
          _bytes_feature(str(metadata[POPULATION_COLUMN])),
      # Nomalize super population to integer or NA_INTEGER if no match.
      'super_population':
          _int64_feature(SUPER_POPULATION_MAP[str(metadata[
              SUPER_POPULATION_COLUMN])]),
      # Use verbatim value of super population.
      'super_population_string':
          _bytes_feature(str(metadata[SUPER_POPULATION_COLUMN])),
      # Nomalize sex/gender to integer or NA_INTEGER if no match.
      'gender':
          _int64_feature(GENDER_MAP[str(metadata[GENDER_COLUMN])]),
      # Use verbatim value of sex/gender.
      'gender_string':
          _bytes_feature(str(metadata[GENDER_COLUMN]))
  }

  for contig, variants in variants_by_contig.iteritems():
    if variants:
      features['variants_' + contig] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=variants))

  return tf.train.Example(features=tf.train.Features(feature=features))


def variants_to_examples(input_data, samples_metadata, add_hethom):
  """Converts variants to TensorFlow Example protos."""
  variant_kvs = input_data | 'bucketVariants' >> beam.Map(
      lambda row: (row[KEY_COLUMN], row))

  sample_variant_kvs = variant_kvs | 'groupBySample' >> beam.GroupByKey()

  examples = sample_variant_kvs | 'samplesToExamples' >> beam.Map(
      lambda (key, vals), samples_metadata: sample_variants_to_example(
          key, vals, samples_metadata, add_hethom),
      AsSingleton(samples_metadata))

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
      '--hethom-words',
      dest='add_hethom',
      action='store_true',
      help='Add variant heterozygous/homozygous "word".'
      'Defaults to true.')
  parser.add_argument(
      '--no-hethom-words',
      dest='add_hethom',
      action='store_false',
      help='Do not add variant heterozygous/homozygous "word".'
      'Defaults to true.')
  parser.set_defaults(add_hethom=True)
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

  metadata_query = str(Template(open(known_args.metadata, 'r').read()).render(
      QUERY_REPLACEMENTS))
  logging.info('metadata query : %s', metadata_query)

  data_query = str(Template(open(known_args.data, 'r').read()).render(
      QUERY_REPLACEMENTS))
  logging.info('data query : %s', data_query)

  beam.coders.registry.register_coder(tf.train.Example, ExampleProtoCoder)
  p = beam.Pipeline(options=pipeline_options)

  # Gather our sample metadata into a python dictionary.
  samples_metadata = (
      p
      | beam.io.Read(
          'readSampleMetadata', beam.io.BigQuerySource(query=metadata_query,
                                                       use_standard_sql=True))
      | 'tableToDictionary'
      >> beam.CombineGlobally(TableToDictCombineFn(key_column=KEY_COLUMN)))

  # Read the table rows into a PCollection.
  rows = p | beam.io.Read(
      'readVariants', beam.io.BigQuerySource(query=data_query,
                                             use_standard_sql=True))

  # Convert the data into TensorFlow Example Protocol Buffers.
  examples = variants_to_examples(
      rows,
      samples_metadata,
      add_hethom=known_args.add_hethom)

  # Write the serialized compressed protocol buffers to Cloud Storage.
  examples | beam.io.Write(
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
