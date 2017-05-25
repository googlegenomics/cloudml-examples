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
"""Test encoding of sample metadata and variant calls to TensorFlow features."""

import copy
import unittest
import trainer.ancestry_metadata_encoder as metadata_encoder
import trainer.feature_encoder as encoder
import trainer.variant_encoder as variant_encoder

# Test data.
SAMPLE_ID = 'sample1'

SAMPLE_METADATA = {}
SAMPLE_METADATA[SAMPLE_ID] = {
    encoder.KEY_COLUMN: SAMPLE_ID,
    metadata_encoder.GENDER_COLUMN: 'female',
    metadata_encoder.SUPER_POPULATION_COLUMN: 'SAS',
    metadata_encoder.POPULATION_COLUMN: 'some pop not in the training labels'
}

HETEROZYGOUS_VARIANT_CALL = {
    encoder.KEY_COLUMN: SAMPLE_ID,
    encoder.CONTIG_COLUMN: 'chr9',
    encoder.START_COLUMN: 3500000,
    encoder.END_COLUMN: 3500001,
    encoder.REF_COLUMN: 'T',
    encoder.ALT_COLUMN: 'G',
    encoder.ALT_NUM_COLUMN: 1,
    encoder.FIRST_ALLELE_COLUMN: 0,
    encoder.SECOND_ALLELE_COLUMN: 1
}

HOMOZYGOUS_ALT_VARIANT_CALL = copy.copy(HETEROZYGOUS_VARIANT_CALL)
HOMOZYGOUS_ALT_VARIANT_CALL[encoder.FIRST_ALLELE_COLUMN] = 1
HOMOZYGOUS_ALT_VARIANT_CALL[encoder.SECOND_ALLELE_COLUMN] = 1

HOMOZYGOUS_REF_VARIANT_CALL = copy.copy(HETEROZYGOUS_VARIANT_CALL)
HOMOZYGOUS_REF_VARIANT_CALL[encoder.FIRST_ALLELE_COLUMN] = 0
HOMOZYGOUS_REF_VARIANT_CALL[encoder.SECOND_ALLELE_COLUMN] = 0


class FeatureEncoderTest(unittest.TestCase):

  def test_sample_to_example(self):
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
    variants_to_features_fn = variant_encoder.build_variants_to_features(
        variant_to_feature_name_fn=variant_encoder.variant_to_contig_feature_name,
        variant_to_words_fn=variant_encoder.build_variant_to_words(
            add_hethom=False))

    sample_to_example = encoder.build_sample_to_example(
        metadata_to_features_fn=metadata_encoder.metadata_to_ancestry_features,
        variants_to_features_fn=variants_to_features_fn)
    self.assertEqual(
        expected,
        str(
            sample_to_example(SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL],
                              SAMPLE_METADATA)))

  def test_sample_to_example_add_hethom(self):
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
    variants_to_features_fn = variant_encoder.build_variants_to_features(
        variant_to_feature_name_fn=variant_encoder.variant_to_contig_feature_name,
        variant_to_words_fn=variant_encoder.build_variant_to_words(
            add_hethom=True))

    sample_to_example = encoder.build_sample_to_example(
        metadata_to_features_fn=metadata_encoder.metadata_to_ancestry_features,
        variants_to_features_fn=variants_to_features_fn)
    self.assertEqual(
        expected,
        str(
            sample_to_example(SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL],
                              SAMPLE_METADATA)))

  def test_sample_to_example_binned_variants(self):
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
    variants_to_features_fn = variant_encoder.build_variants_to_features(
        variant_to_feature_name_fn=variant_encoder.
        build_variant_to_binned_feature_name(),
        variant_to_words_fn=variant_encoder.build_variant_to_words(
            add_hethom=False))

    sample_to_example = encoder.build_sample_to_example(
        metadata_to_features_fn=metadata_encoder.metadata_to_ancestry_features,
        variants_to_features_fn=variants_to_features_fn)
    self.assertEqual(
        expected,
        str(
            sample_to_example(SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL],
                              SAMPLE_METADATA)))

  def test_sample_to_example_smaller_bins(self):
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
    variants_to_features_fn = variant_encoder.build_variants_to_features(
        variant_to_feature_name_fn=variant_encoder.
        build_variant_to_binned_feature_name(bin_size=100000),
        variant_to_words_fn=variant_encoder.build_variant_to_words(
            add_hethom=False))

    sample_to_example = encoder.build_sample_to_example(
        metadata_to_features_fn=metadata_encoder.metadata_to_ancestry_features,
        variants_to_features_fn=variants_to_features_fn)
    self.assertEqual(
        expected,
        str(
            sample_to_example(SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL],
                              SAMPLE_METADATA)))

  def test_sample_to_example_binned_variants_add_hethom(self):
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
    variants_to_features_fn = variant_encoder.build_variants_to_features(
        variant_to_feature_name_fn=variant_encoder.
        build_variant_to_binned_feature_name(),
        variant_to_words_fn=variant_encoder.build_variant_to_words(
            add_hethom=True))

    sample_to_example = encoder.build_sample_to_example(
        metadata_to_features_fn=metadata_encoder.metadata_to_ancestry_features,
        variants_to_features_fn=variants_to_features_fn)
    self.assertEqual(
        expected,
        str(
            sample_to_example(SAMPLE_ID, [HETEROZYGOUS_VARIANT_CALL],
                              SAMPLE_METADATA)))


if __name__ == '__main__':
  unittest.main()
