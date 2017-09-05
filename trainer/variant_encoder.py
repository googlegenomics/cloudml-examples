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
"""Encode variant calls as TensorFlow features."""

import collections
import struct

import farmhash
import tensorflow as tf
import trainer.feature_encoder as encoder
import trainer.util as util

# Normalize reference names, defaulting to the vertabim reference name
# if value is not present in the dictionary.
CONTIG_MAP = util.DefaultToKeyDict()
CONTIG_MAP.update({('chr' + str(x)): str(x) for x in range(1, 23)})
CONTIG_MAP['chrX'] = 'X'
CONTIG_MAP['chrY'] = 'Y'
CONTIG_MAP['chrM'] = 'MT'
CONTIG_MAP['chrMT'] = 'MT'
CONTIG_MAP['M'] = 'MT'


def normalize_contig_name(variant):
  """Normalize reference (contig) names.

  For example chromosome X might be 'X' in one dataset and 'chrX' in
  another.

  Args:
    variant: a variant call

  Returns:
    The canonical name of the reference (contig) specified in the
    variant call.
  """
  return CONTIG_MAP[variant[encoder.CONTIG_COLUMN]]


def variant_to_contig_feature_name(variant):
  """Create the feature name for the variant call.

  In this implementation the feature name is merly the name of the
  reference (contig) within the variant call.

  Args:
    variant: a variant call

  Returns:
    The name for the feature in which this variant should be stored.
  """
  # Use normalized reference name as feature name.
  return normalize_contig_name(variant)


def sample_has_variant(variant):
  """Check whether the sample has this particular variant.

  Since the input data was FLATTENED on alternate_bases, we do this by
  checking whether either allele value corresponds to the alternate
  currently under consideration. Note that the values of the first
  allele and the second allele are genotypes --> which are essentially
  an index into the alternate_bases repeated field.  See
  http://vcftools.sourceforge.net/VCF-poster.pdf for more detail.

  Args:
    variant: a variant call

  Returns:
    A count of the alleles for this alternate. This count can also be
      interpreted as a boolean to indicate whether or not the sample
      has this particular variant allele.
  """
  alt_num = int(variant[encoder.ALT_NUM_COLUMN])
  return ((variant[encoder.FIRST_ALLELE_COLUMN] == alt_num) +
          (encoder.SECOND_ALLELE_COLUMN in variant and
           variant[encoder.SECOND_ALLELE_COLUMN] == alt_num))


def build_variant_to_binned_feature_name(bin_size=1000000):
  """Builder for strategy for separate features for contiguous genomic regions.

  Args:
    bin_size: the maximum size of each contiguous genomic region.

  Returns:
    The instantiated strategy.
  """

  def variant_to_binned_feature_name(variant):
    """Create the feature name for the variant call.

    In this implementation the feature name is merly the name of the
    reference (contig) within the variant call contenated with the bin
    number of the genomic region in which the variant call resides.

    Args:
      variant: a variant call

    Returns:
      The name for the feature in which this variant call should be stored.
    """
    # variant_bin will be the floor result of the division since bin_size
    # is an integer.
    variant_bin = int(variant[encoder.START_COLUMN]) / bin_size
    return '_'.join([normalize_contig_name(variant), str(variant_bin)])

  return variant_to_binned_feature_name


def build_variant_to_words(add_hethom=True):
  """Builder for strategy to convert a variant to words.

  This encoder will create separate bag-of-words features for each
  reference name (contig) in the source data.

  Args:
    add_hethom: whether or not to add additional words representing
      the zygosity of the variant call.

  Returns:
    The instantiated strategy.
  """

  def variant_to_words(variant):
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
    # Only add words only if the sample has a variant at this site.
    if not sample_has_variant(variant):
      return []

    # Normalize reference names in the words.
    contig = normalize_contig_name(variant)

    variant_word = '_'.join([
        contig,
        str(variant[encoder.START_COLUMN]),
        str(variant[encoder.END_COLUMN]),
        str(variant[encoder.REF_COLUMN]),
        str(variant[encoder.ALT_COLUMN])
    ])

    if not add_hethom:
      return [variant_word]

    # Add two words, one for the variant itself and another more specific word
    # (a synonym) regarding heterozygosity/homozygosity of the observation.
    if ((encoder.SECOND_ALLELE_COLUMN not in variant) or
        (variant[encoder.FIRST_ALLELE_COLUMN] !=
         variant[encoder.SECOND_ALLELE_COLUMN])):
      return [variant_word, '_'.join([variant_word, 'het'])]

    return [variant_word, '_'.join([variant_word, 'hom'])]

  return variant_to_words


def build_variants_to_features(variant_to_feature_name_fn, variant_to_words_fn):
  """Builder for the strategy to convert variants to bag-of-words features.

  Args:
    variant_to_feature_name_fn: strategy to determine the feature name (bag
      name) from the variant
    variant_to_words_fn: strategy to encode the variant as one or more words

  Returns:
    The instantiated strategy.
  """

  def variants_to_features(sample_variants):
    """Convert variant calls to TensorFlow features.

    See also
    https://www.tensorflow.org/versions/r0.10/how_tos/reading_data/index.html

    Args:
      sample_variants: the sample's variant calls

    Returns:
      A dictionary of TensorFlow features.
    """
    variants_by_feature = collections.defaultdict(list)
    for variant in sample_variants:
      feature_name = variant_to_feature_name_fn(variant)
      words = variant_to_words_fn(variant)
      variants_by_feature[feature_name].extend(
          # fingerprint64 returns an unsigned int64 but int64 features are
          # signed.  Convert from from unsigned to signed.
          [
              struct.unpack('q', struct.pack('Q', farmhash.fingerprint64(w)))[0]
              for w in words
          ])

    # Fill in features from variants.
    features = {}
    for feature, variants in variants_by_feature.iteritems():
      if variants:
        features['variants_' + feature] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=variants))

    return features

  return variants_to_features
