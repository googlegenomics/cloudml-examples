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
"""Several encodings to transform variant calls to TensorFlow features."""

import collections
from collections import defaultdict
import math
import struct

import farmhash
import tensorflow as tf
import trainer.util as util


class FeatureEncoder(object):
  """Encode variant calls as TensorFlow features."""

  # Values to use for missing data.
  NA_STRING = 'NA'
  NA_INTEGER = -1

  # Decouple source table column names from the dictionary keys used
  # in this code.
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

  # Normalize reference names, defaulting to the vertabim reference name
  # if value is not present in the dictionary.
  CONTIG_MAP = util.DefaultToKeyDict()
  CONTIG_MAP.update({('chr' + str(x)): str(x) for x in range(1, 23)})
  CONTIG_MAP['chrX'] = 'X'
  CONTIG_MAP['chrY'] = 'Y'
  CONTIG_MAP['chrM'] = 'MT'
  CONTIG_MAP['chrMT'] = 'MT'
  CONTIG_MAP['M'] = 'MT'

  # Normalize over possible sex/gender values.
  GENDER_MAP = defaultdict(lambda: FeatureEncoder.NA_INTEGER)
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

  # Population and Super population labels used for training and evaluation
  # will always be the ones from 1000 Genomes. If we wish to use another
  # dataset for training and evaluation, we'll need to provide the mapping
  # from 1000 Genomes labels to those used for the dataset.
  SUPER_POPULATIONS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']

  SUPER_POPULATION_MAP = defaultdict(lambda: FeatureEncoder.NA_INTEGER)
  for pop in range(len(SUPER_POPULATIONS)):
    SUPER_POPULATION_MAP[SUPER_POPULATIONS[pop]] = pop

  POPULATIONS = [
      'ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN',
      'GBR', 'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL',
      'PEL', 'PJL', 'PUR', 'STU', 'TSI', 'YRI'
  ]

  POPULATION_MAP = defaultdict(lambda: FeatureEncoder.NA_INTEGER)
  for pop in range(len(POPULATIONS)):
    POPULATION_MAP[POPULATIONS[pop]] = pop

  def __init__(self, add_hethom=True):
    """Initializes ``FeatureEncoder``.

    This encoder will create separate bag-of-words features for each
    reference name (contig) in the source data.

    Args:
      add_hethom: whether or not to add additional words representing
        the zygosity of the variant call.
    """
    self.add_hethom = add_hethom

  def normalize_contig_name(self, variant):
    """Normalize reference (contig) names.

    For example chromosome X might be 'X' in one dataset and 'chrX' in
    another.

    Args:
      variant: a variant call

    Returns:
      The canonical name of the reference (contig) specified in the
      variant call.
    """
    return self.CONTIG_MAP[variant[self.CONTIG_COLUMN]]

  def variant_to_feature_name(self, variant):
    """Create the feature name for the variant call.

    In this implementation the feature name is merly the name of the
    reference (contig) within the variant call.

    Args:
      variant: a variant call

    Returns:
      The name for the feature in which this variant should be stored.
    """
    # Use normalized reference name as feature name.
    return self.normalize_contig_name(variant)

  def variant_to_words(self, variant):
    """Encode a variant call as one or more "words".

    Given a variant call record with certain expected fields, create
    "words" that uniquely describe it. The first word would match
    both heterozygous or homozygous variant calls.  A second word is
    created when add_hethom=True and is more specific, matching just
    one of heterozygous or homozygous.

    Args:
      variant: a variant call

    Returns:
      One or more "words" that represent the variant call.
    """
    # Only add words only if the sample has a variant at this site. Since the
    # input data was FLATTENED on alternate_bases, we do this by checking
    # whether either allele value corresponds to the alternate currently
    # under consideration. Note that the values of the first allele and
    # the second allele are genotypes --> which are essentially an index
    # into the alternate_bases repeated field.
    # See http://vcftools.sourceforge.net/VCF-poster.pdf for more detail.
    alt_num = int(variant[self.ALT_NUM_COLUMN])
    if ((variant[self.FIRST_ALLELE_COLUMN] != alt_num) and (
        (self.SECOND_ALLELE_COLUMN not in variant) or
        (variant[self.SECOND_ALLELE_COLUMN] != alt_num))):
      return []

    # Normalize reference names in the words.
    contig = self.normalize_contig_name(variant)

    variant_word = '_'.join([
        contig, str(variant[self.START_COLUMN]), str(variant[self.END_COLUMN]),
        str(variant[self.REF_COLUMN]), str(variant[self.ALT_COLUMN])
    ])

    if not self.add_hethom:
      return [variant_word]

    # Add two words, one for the variant itself and another more specific word
    # (a synonym) regarding heterozygosity/homozygosity of the observation.
    if ((self.SECOND_ALLELE_COLUMN not in variant) or
        (variant[self.FIRST_ALLELE_COLUMN] != variant[self.SECOND_ALLELE_COLUMN]
        )):
      return [variant_word, '_'.join([variant_word, 'het'])]

    return [variant_word, '_'.join([variant_word, 'hom'])]

  def sample_variants_to_example(self, sample, sample_variants,
                                 samples_metadata):
    """Convert variant calls to TensorFlow Example protocol buffers.

    See also
    https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html

    Args:
      sample: the identifier for the sample
      sample_variants: the sample's variant calls
      samples_metadata: dictionary of metadata for all samples

    Returns:
      A filled in TensorFlow Example proto for this sample.
    """
    # Some samples may have no metadata, but we may still want to preprocess
    # the data for prediction use cases.
    metadata = defaultdict(lambda: self.NA_STRING)
    if sample in samples_metadata:
      metadata.update(samples_metadata[sample])

    variants_by_feature = collections.defaultdict(list)
    for variant in sample_variants:
      feature_name = self.variant_to_feature_name(variant)
      words = self.variant_to_words(variant)
      variants_by_feature[feature_name].extend(
          # fingerprint64 returns an unsigned int64 but int64 features are
          # signed.  Convert from from unsigned to signed.
          [
              struct.unpack('q', struct.pack('Q', farmhash.fingerprint64(w)))[0]
              for w in words
          ])

    features = {
        'sample_name':
            util.bytes_feature(str(sample)),
        # Nomalize population to integer or NA_INTEGER if no match.
        'population':
            util.int64_feature(self.POPULATION_MAP[str(metadata[
                self.POPULATION_COLUMN])]),
        # Use verbatim value of population.
        'population_string':
            util.bytes_feature(str(metadata[self.POPULATION_COLUMN])),
        # Nomalize super population to integer or NA_INTEGER if no match.
        'super_population':
            util.int64_feature(self.SUPER_POPULATION_MAP[str(metadata[
                self.SUPER_POPULATION_COLUMN])]),
        # Use verbatim value of super population.
        'super_population_string':
            util.bytes_feature(str(metadata[self.SUPER_POPULATION_COLUMN])),
        # Nomalize sex/gender to integer or NA_INTEGER if no match.
        'gender':
            util.int64_feature(self.GENDER_MAP[str(metadata[
                self.GENDER_COLUMN])]),
        # Use verbatim value of sex/gender.
        'gender_string':
            util.bytes_feature(str(metadata[self.GENDER_COLUMN]))
    }

    for feature, variants in variants_by_feature.iteritems():
      if variants:
        features['variants_' + feature] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=variants))

    return tf.train.Example(features=tf.train.Features(feature=features))


class BinnedFeatureEncoder(FeatureEncoder):
  """Encode variant calls as TensorFlow features.

  This derived class uses a different strategy than the parent for allocating
  encoded variants to TensorFlow features.  Specifically, it groups together
  variant calls within the same genomic region.
  """

  def __init__(self, add_hethom=True, bin_size=1000000):
    """Initializes ``BinnedFeatureEncoder``.

    This encoder will create separate bag-of-words features for contiguous
    genomic regions within each reference name (contig) in the source data.

    Args:
      add_hethom: whether or not to add additional words representing
        the zygosity of the variant call.
      bin_size: the maximum size of each contiguous genomic region.
    """
    super(BinnedFeatureEncoder, self).__init__(add_hethom)
    self.bin_size = bin_size

  def variant_to_feature_name(self, variant):
    """Create the feature name for the variant call.

    In this implementation the feature name is merly the name of the
    reference (contig) within the variant call contenated with the bin
    number of the genomic region in which the variant call resides.

    Args:
      variant: a variant call

    Returns:
      The name for the feature in which this variant call should be stored.
    """
    variant_bin = int(
        math.floor(int(variant[self.START_COLUMN]) / self.bin_size))
    return '_'.join([self.normalize_contig_name(variant), str(variant_bin)])
