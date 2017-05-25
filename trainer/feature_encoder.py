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
"""Encode sample metadata and variant calls as TensorFlow features.

Given sample metadata and variants, the sample_to_example method will
call the specified encoding strategies to construct the final TensorFlow
Example protocol buffer.
"""

from collections import defaultdict

import tensorflow as tf

import trainer.util as util

# Values to use for missing data.
NA_STRING = 'NA'
NA_INTEGER = -1

# Decouple variant data source table column names from the dictionary
# keys used in the variant encoders.
KEY_COLUMN = 'key'
CONTIG_COLUMN = 'contig'
START_COLUMN = 'start_pos'
END_COLUMN = 'end_pos'
REF_COLUMN = 'ref'
ALT_COLUMN = 'alt'
ALT_NUM_COLUMN = 'alt_num'
FIRST_ALLELE_COLUMN = 'first'
SECOND_ALLELE_COLUMN = 'second'

# Feature name constants
SAMPLE_NAME_FEATURE = 'sample_name'


def build_sample_to_example(metadata_to_features_fn, variants_to_features_fn):
  """Builder for the strategy to construct examples from sample data.

  Args:
    metadata_to_features_fn: the strategy to encode sample metadata as features
    variants_to_features_fn: the strategy to encode sample variants as features

  Returns:
    The instantiated strategy.
  """

  def sample_to_example(sample, sample_variants, samples_metadata):
    """Convert sample metadata and variants to TensorFlow examples.

    Args:
      sample: the identifier for the sample
      sample_variants: the sample's variant calls
      samples_metadata: dictionary of metadata for all samples

    Returns:
      A filled in TensorFlow Example protocol buffer for this sample.
    """
    features = {SAMPLE_NAME_FEATURE: util.bytes_feature(str(sample))}

    # Some samples may have no metadata, but we may still want to preprocess
    # the data for prediction use cases.
    metadata = defaultdict(lambda: NA_STRING)
    if sample in samples_metadata:
      metadata.update(samples_metadata[sample])

    # Fill in features from metadata.
    features.update(metadata_to_features_fn(metadata))

    # Fill in features from variants.
    features.update(variants_to_features_fn(sample_variants))

    return tf.train.Example(features=tf.train.Features(feature=features))

  return sample_to_example
