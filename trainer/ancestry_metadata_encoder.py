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
"""Encode sample metadata as features for an ancestry inference."""

from collections import defaultdict

import trainer.feature_encoder as encoder
import trainer.util as util

# Decouple source table column names from the dictionary keys used
# in this code.
POPULATION_COLUMN = 'pop'
SUPER_POPULATION_COLUMN = 'sup'
GENDER_COLUMN = 'sex'

# Normalize over possible sex/gender values.
GENDER_MAP = defaultdict(lambda: encoder.NA_INTEGER)
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
# will always be the ones from 1000 Genomes plus a label for unknown
# populations. If we wish to use another dataset for training and
# evaluation, we'll need to provide the mapping from 1000 Genomes
# labels to those used for the dataset.
SUPER_POPULATIONS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS', 'UNK']

SUPER_POPULATION_MAP = defaultdict(lambda: encoder.NA_INTEGER)
for pop in range(len(SUPER_POPULATIONS)):
  SUPER_POPULATION_MAP[SUPER_POPULATIONS[pop]] = pop

POPULATIONS = [
    'ACB', 'ASW', 'BEB', 'CDX', 'CEU', 'CHB', 'CHS', 'CLM', 'ESN', 'FIN', 'GBR',
    'GIH', 'GWD', 'IBS', 'ITU', 'JPT', 'KHV', 'LWK', 'MSL', 'MXL', 'PEL', 'PJL',
    'PUR', 'STU', 'TSI', 'YRI'
]

POPULATION_MAP = defaultdict(lambda: encoder.NA_INTEGER)
for pop in range(len(POPULATIONS)):
  POPULATION_MAP[POPULATIONS[pop]] = pop

# Metadata feature name constants
POPULATION_FEATURE = 'population'
POPULATION_STRING_FEATURE = 'population_string'
SUPER_POPULATION_FEATURE = 'super_population'
SUPER_POPULATION_STRING_FEATURE = 'super_population_string'
GENDER_FEATURE = 'gender'
GENDER_STRING_FEATURE = 'gender_string'


def metadata_to_ancestry_features(sample_metadata):
  """Create features from sample metadata.

  Args:
      sample_metadata: dictionary of metadata for one sample

  Returns:
      A dictionary of TensorFlow features.
  """
  features = {
      # Nomalize population to integer or NA_INTEGER if no match.
      POPULATION_FEATURE:
          util.int64_feature(
              POPULATION_MAP[str(sample_metadata[POPULATION_COLUMN])]),
      # Use verbatim value of population.
      POPULATION_STRING_FEATURE:
          util.bytes_feature(str(sample_metadata[POPULATION_COLUMN])),
      # Nomalize super population to integer or NA_INTEGER if no match.
      SUPER_POPULATION_FEATURE:
          util.int64_feature(SUPER_POPULATION_MAP[str(sample_metadata[
              SUPER_POPULATION_COLUMN])]),
      # Use verbatim value of super population.
      SUPER_POPULATION_STRING_FEATURE:
          util.bytes_feature(str(sample_metadata[SUPER_POPULATION_COLUMN])),
      # Nomalize sex/gender to integer or NA_INTEGER if no match.
      GENDER_FEATURE:
          util.int64_feature(GENDER_MAP[str(sample_metadata[GENDER_COLUMN])]),
      # Use verbatim value of sex/gender.
      GENDER_STRING_FEATURE:
          util.bytes_feature(str(sample_metadata[GENDER_COLUMN]))
  }
  return features
