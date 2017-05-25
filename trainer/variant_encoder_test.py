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

import trainer.feature_encoder as encoder
import trainer.variant_encoder as variant_encoder

SAMPLE_ID = 'sample1'

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


class VariantEncoderTest(unittest.TestCase):

  def test_normalize_autosome_contig_names(self):
    self.assertEqual('1',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: '1'
                     }))
    self.assertEqual('1',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'chr1'
                     }))
    self.assertEqual('21',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: '21'
                     }))
    self.assertEqual('21',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'chr21'
                     }))

  def test_normalize_sex_contig_names(self):
    self.assertEqual('X',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'X'
                     }))
    self.assertEqual('X',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'chrX'
                     }))
    self.assertEqual('Y',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'Y'
                     }))
    self.assertEqual('Y',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'chrY'
                     }))

  def test_normalize_mitochondrial_contig_names(self):
    self.assertEqual('MT',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'MT'
                     }))
    self.assertEqual('MT',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'M'
                     }))
    self.assertEqual('MT',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'chrM'
                     }))
    self.assertEqual('MT',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'chrMT'
                     }))

  def test_normalize_other_contig_names(self):
    # All others pass through as-is.
    self.assertEqual('KI270375.1',
                     variant_encoder.normalize_contig_name({
                         encoder.CONTIG_COLUMN: 'KI270375.1'
                     }))

  def test_variant_to_feature_name(self):
    self.assertEqual('9',
                     variant_encoder.variant_to_contig_feature_name({
                         encoder.CONTIG_COLUMN: 'chr9',
                         encoder.START_COLUMN: 3500000
                     }))

  def test_variant_to_binned_feature_name(self):
    variant_to_feature_name_fn = (
        variant_encoder.build_variant_to_binned_feature_name())
    self.assertEqual('9_3',
                     variant_to_feature_name_fn({
                         encoder.CONTIG_COLUMN: 'chr9',
                         encoder.START_COLUMN: 3500000
                     }))
    self.assertEqual('9_4',
                     variant_to_feature_name_fn({
                         encoder.CONTIG_COLUMN: 'chr9',
                         encoder.START_COLUMN: 4000000
                     }))

  def test_variant_to_smaller_binned_feature_name(self):
    variant_to_feature_name_fn = (
        variant_encoder.build_variant_to_binned_feature_name(bin_size=100000))
    self.assertEqual('9_35',
                     variant_to_feature_name_fn({
                         encoder.CONTIG_COLUMN: 'chr9',
                         encoder.START_COLUMN: 3500000
                     }))
    self.assertEqual('9_40',
                     variant_to_feature_name_fn({
                         encoder.CONTIG_COLUMN: 'chr9',
                         encoder.START_COLUMN: 4000000
                     }))

  def test_variant_to_words(self):
    variant_to_words_fn = variant_encoder.build_variant_to_words(
        add_hethom=False)
    self.assertEqual(['9_3500000_3500001_T_G'],
                     variant_to_words_fn(HETEROZYGOUS_VARIANT_CALL))
    self.assertEqual(['9_3500000_3500001_T_G'],
                     variant_to_words_fn(HOMOZYGOUS_ALT_VARIANT_CALL))
    self.assertEqual([], variant_to_words_fn(HOMOZYGOUS_REF_VARIANT_CALL))

  def test_variant_to_words_add_het_hom(self):
    variant_to_words_fn = variant_encoder.build_variant_to_words()
    self.assertEqual(['9_3500000_3500001_T_G', '9_3500000_3500001_T_G_het'],
                     variant_to_words_fn(HETEROZYGOUS_VARIANT_CALL))
    self.assertEqual(['9_3500000_3500001_T_G', '9_3500000_3500001_T_G_hom'],
                     variant_to_words_fn(HOMOZYGOUS_ALT_VARIANT_CALL))
    self.assertEqual([], variant_to_words_fn(HOMOZYGOUS_REF_VARIANT_CALL))


if __name__ == '__main__':
  unittest.main()
