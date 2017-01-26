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
"""Reusable utility functions.

This file is generic and can be reused by other models without modification.
"""

from apache_beam.transforms import core
import tensorflow as tf


def int64_feature(value):
  """Create a multi-valued int64 feature from a single value."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  """Create a multi-valued bytes feature from a single value."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  """Create a multi-valued float feature from a single value."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class DefaultToKeyDict(dict):
  """Custom dictionary to use the key as the value for any missing entries."""

  def __missing__(self, key):
    return str(key)


class TableToDictCombineFn(core.CombineFn):
  """Beam transform to create a python dictionary from a BigQuery table.

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
