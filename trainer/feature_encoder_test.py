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
"""Test encoding of variant calls to TensorFlow features."""

import copy
import unittest
from trainer.feature_encoder import BinnedFeatureEncoder
from trainer.feature_encoder import FeatureEncoder

# Test data.
SAMPLE_ID = 'sample1'

SAMPLE_METADATA = {}
SAMPLE_METADATA[SAMPLE_ID] = {
    FeatureEncoder.KEY_COLUMN: SAMPLE_ID,
    FeatureEncoder.GENDER_COLUMN: 'female',
    FeatureEncoder.SUPER_POPULATION_COLUMN: 'SAS',
    FeatureEncoder.POPULATION_COLUMN: 'some pop not in the training labels'
}

HETEROZYGOUS_VARIANT_CALL = {
    FeatureEncoder.KEY_COLUMN: SAMPLE_ID,
    FeatureEncoder.CONTIG_COLUMN: 'chr9',
    FeatureEncoder.START_COLUMN: 3500000,
    FeatureEncoder.END_COLUMN: 3500001,
    FeatureEncoder.REF_COLUMN: 'T',
    FeatureEncoder.ALT_COLUMN: 'G',
    FeatureEncoder.ALT_NUM_COLUMN: 1,
    FeatureEncoder.FIRST_ALLELE_COLUMN: 0,
    FeatureEncoder.SECOND_ALLELE_COLUMN: 1
}

HOMOZYGOUS_ALT_VARIANT_CALL = copy.copy(HETEROZYGOUS_VARIANT_CALL)
HOMOZYGOUS_ALT_VARIANT_CALL[FeatureEncoder.FIRST_ALLELE_COLUMN] = 1
HOMOZYGOUS_ALT_VARIANT_CALL[FeatureEncoder.SECOND_ALLELE_COLUMN] = 1

HOMOZYGOUS_REF_VARIANT_CALL = copy.copy(HETEROZYGOUS_VARIANT_CALL)
HOMOZYGOUS_REF_VARIANT_CALL[FeatureEncoder.FIRST_ALLELE_COLUMN] = 0
HOMOZYGOUS_REF_VARIANT_CALL[FeatureEncoder.SECOND_ALLELE_COLUMN] = 0


class FeatureEncodingTest(unittest.TestCase):

  def setUp(self):
    self.encoder = FeatureEncoder(add_hethom=False)

  def test_normalize_autosome_contig_names(self):
    self.assertEqual('1',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: '1'
                     }))
    self.assertEqual('1',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr1'
                     }))
    self.assertEqual('21',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: '21'
                     }))
    self.assertEqual('21',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr21'
                     }))

  def test_normalize_sex_contig_names(self):
    self.assertEqual('X',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'X'
                     }))
    self.assertEqual('X',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chrX'
                     }))
    self.assertEqual('Y',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'Y'
                     }))
    self.assertEqual('Y',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chrY'
                     }))

  def test_normalize_mitochondrial_contig_names(self):
    self.assertEqual('MT',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'MT'
                     }))
    self.assertEqual('MT',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'M'
                     }))
    self.assertEqual('MT',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chrM'
                     }))
    self.assertEqual('MT',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chrMT'
                     }))

  def test_normalize_other_contig_names(self):
    # All others pass through as-is.
    self.assertEqual('KI270375.1',
                     self.encoder.normalize_contig_name({
                         FeatureEncoder.CONTIG_COLUMN: 'KI270375.1'
                     }))

  def test_variant_to_feature_name(self):
    self.assertEqual('9',
                     self.encoder.variant_to_feature_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr9',
                         FeatureEncoder.START_COLUMN: 3500000
                     }))

  def test_variant_to_words_heterozygous(self):
    self.assertEqual(['9_3500000_3500001_T_G'],
                     self.encoder.variant_to_words(HETEROZYGOUS_VARIANT_CALL))

  def test_variant_to_words_homozygous_alt(self):
    self.assertEqual(['9_3500000_3500001_T_G'],
                     self.encoder.variant_to_words(HOMOZYGOUS_ALT_VARIANT_CALL))

  def test_variant_to_words_homozygous_ref(self):
    self.assertEqual([],
                     self.encoder.variant_to_words(HOMOZYGOUS_REF_VARIANT_CALL))

  def test_sample_variants_to_example(self):
    expected = """features {
  feature {
    key: "gender"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "gender_string"
    value {
      bytes_list {
        value: "female"
      }
    }
  }
  feature {
    key: "population"
    value {
      int64_list {
        value: -1
      }
    }
  }
  feature {
    key: "population_string"
    value {
      bytes_list {
        value: "some pop not in the training labels"
      }
    }
  }
  feature {
    key: "sample_name"
    value {
      bytes_list {
        value: "sample1"
      }
    }
  }
  feature {
    key: "super_population"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "super_population_string"
    value {
      bytes_list {
        value: "SAS"
      }
    }
  }
  feature {
    key: "variants_9"
    value {
      int64_list {
        value: -5153783975271321865
      }
    }
  }
}
"""

    self.assertEqual(
        expected,
        str(
            self.encoder.sample_variants_to_example(
                SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL], SAMPLE_METADATA)))


class HetHomFeatureEncodingTest(FeatureEncodingTest):

  def setUp(self):
    self.encoder = FeatureEncoder(add_hethom=True)

  def test_variant_to_words_heterozygous(self):
    self.assertEqual(['9_3500000_3500001_T_G', '9_3500000_3500001_T_G_het'],
                     self.encoder.variant_to_words(HETEROZYGOUS_VARIANT_CALL))

  def test_variant_to_words_homozygous_alt(self):
    self.assertEqual(['9_3500000_3500001_T_G', '9_3500000_3500001_T_G_hom'],
                     self.encoder.variant_to_words(HOMOZYGOUS_ALT_VARIANT_CALL))

  def test_sample_variants_to_example(self):
    expected = """features {
  feature {
    key: "gender"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "gender_string"
    value {
      bytes_list {
        value: "female"
      }
    }
  }
  feature {
    key: "population"
    value {
      int64_list {
        value: -1
      }
    }
  }
  feature {
    key: "population_string"
    value {
      bytes_list {
        value: "some pop not in the training labels"
      }
    }
  }
  feature {
    key: "sample_name"
    value {
      bytes_list {
        value: "sample1"
      }
    }
  }
  feature {
    key: "super_population"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "super_population_string"
    value {
      bytes_list {
        value: "SAS"
      }
    }
  }
  feature {
    key: "variants_9"
    value {
      int64_list {
        value: -5153783975271321865
        value: 1206215103517908850
      }
    }
  }
}
"""
    self.assertEqual(
        expected,
        str(
            self.encoder.sample_variants_to_example(
                SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL], SAMPLE_METADATA)))


class BinnedFeatureEncodingTest(FeatureEncodingTest):

  def setUp(self):
    self.encoder = BinnedFeatureEncoder(add_hethom=False)

  def test_variant_to_feature_name(self):
    self.assertEqual('9_3',
                     self.encoder.variant_to_feature_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr9',
                         FeatureEncoder.START_COLUMN: 3500000
                     }))
    self.assertEqual('9_4',
                     self.encoder.variant_to_feature_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr9',
                         FeatureEncoder.START_COLUMN: 4000000
                     }))

  def test_sample_variants_to_example(self):
    expected = """features {
  feature {
    key: "gender"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "gender_string"
    value {
      bytes_list {
        value: "female"
      }
    }
  }
  feature {
    key: "population"
    value {
      int64_list {
        value: -1
      }
    }
  }
  feature {
    key: "population_string"
    value {
      bytes_list {
        value: "some pop not in the training labels"
      }
    }
  }
  feature {
    key: "sample_name"
    value {
      bytes_list {
        value: "sample1"
      }
    }
  }
  feature {
    key: "super_population"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "super_population_string"
    value {
      bytes_list {
        value: "SAS"
      }
    }
  }
  feature {
    key: "variants_9_3"
    value {
      int64_list {
        value: -5153783975271321865
      }
    }
  }
}
"""
    self.assertEqual(
        expected,
        str(
            self.encoder.sample_variants_to_example(
                SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL], SAMPLE_METADATA)))


class SmallerBinnedFeatureEncodingTest(BinnedFeatureEncodingTest):

  def setUp(self):
    self.encoder = BinnedFeatureEncoder(add_hethom=False, bin_size=100000)

  def test_variant_to_feature_name(self):
    self.assertEqual('9_35',
                     self.encoder.variant_to_feature_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr9',
                         FeatureEncoder.START_COLUMN: 3500000
                     }))
    self.assertEqual('9_40',
                     self.encoder.variant_to_feature_name({
                         FeatureEncoder.CONTIG_COLUMN: 'chr9',
                         FeatureEncoder.START_COLUMN: 4000000
                     }))

  def test_sample_variants_to_example(self):
    expected = """features {
  feature {
    key: "gender"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "gender_string"
    value {
      bytes_list {
        value: "female"
      }
    }
  }
  feature {
    key: "population"
    value {
      int64_list {
        value: -1
      }
    }
  }
  feature {
    key: "population_string"
    value {
      bytes_list {
        value: "some pop not in the training labels"
      }
    }
  }
  feature {
    key: "sample_name"
    value {
      bytes_list {
        value: "sample1"
      }
    }
  }
  feature {
    key: "super_population"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "super_population_string"
    value {
      bytes_list {
        value: "SAS"
      }
    }
  }
  feature {
    key: "variants_9_35"
    value {
      int64_list {
        value: -5153783975271321865
      }
    }
  }
}
"""
    self.assertEqual(
        expected,
        str(
            self.encoder.sample_variants_to_example(
                SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL], SAMPLE_METADATA)))


class HetHomBinnedFeatureEncodingTest(BinnedFeatureEncodingTest):

  def setUp(self):
    self.encoder = BinnedFeatureEncoder(add_hethom=True)

  def test_variant_to_words_heterozygous(self):
    self.assertEqual(['9_3500000_3500001_T_G', '9_3500000_3500001_T_G_het'],
                     self.encoder.variant_to_words(HETEROZYGOUS_VARIANT_CALL))

  def test_variant_to_words_homozygous_alt(self):
    self.assertEqual(['9_3500000_3500001_T_G', '9_3500000_3500001_T_G_hom'],
                     self.encoder.variant_to_words(HOMOZYGOUS_ALT_VARIANT_CALL))

  def test_sample_variants_to_example(self):
    expected = """features {
  feature {
    key: "gender"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "gender_string"
    value {
      bytes_list {
        value: "female"
      }
    }
  }
  feature {
    key: "population"
    value {
      int64_list {
        value: -1
      }
    }
  }
  feature {
    key: "population_string"
    value {
      bytes_list {
        value: "some pop not in the training labels"
      }
    }
  }
  feature {
    key: "sample_name"
    value {
      bytes_list {
        value: "sample1"
      }
    }
  }
  feature {
    key: "super_population"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "super_population_string"
    value {
      bytes_list {
        value: "SAS"
      }
    }
  }
  feature {
    key: "variants_9_3"
    value {
      int64_list {
        value: -5153783975271321865
        value: 1206215103517908850
      }
    }
  }
}
"""
    self.assertEqual(
        expected,
        str(
            self.encoder.sample_variants_to_example(
                SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL], SAMPLE_METADATA)))


if __name__ == '__main__':
  unittest.main()
