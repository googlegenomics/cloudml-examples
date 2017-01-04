#!/usr/bin/python

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Extract embedding values from the specified model.

Usage:
extract_embeddings \
  --model_path <PATH_TO_MODEL> \
  --input_path '<DATA_PATH_1>/examples-*,<DATA_PATH_2>/examples*' \
  --fetch embeddings \
  --output_path <OUTPUT_PATH>/population_embeddings.tsv
"""

import json
import sys

import tensorflow as tf

from tensorflow.contrib.session_bundle import session_bundle
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


tf.flags.DEFINE_string('model_path', None, 'Path for the model.')
tf.flags.DEFINE_string(
    'input_path', None, 'Path to the input tf.Example records.')
tf.flags.DEFINE_string(
    'output_path', None, 'Path for the output tsv.')
tf.flags.DEFINE_string(
    'fetch', None, 'Type ouf output to fetch, one of "score" or "embeddings"')
tf.flags.DEFINE_integer('offset', 0, 'Dataset offset')

FLAGS = tf.flags.FLAGS


def extract_embeddings():
  """Extract embedding vectors from the specified model and paths."""
  session, _ = session_bundle.load_session_bundle_from_path(FLAGS.model_path)
  all_paths = FLAGS.input_path.split(',')
  with tf.gfile.Open(FLAGS.output_path, 'w') as out:
    for dataset, pattern in enumerate(all_paths, start=FLAGS.offset):
      paths = tf.gfile.Glob(pattern)
      for path in paths:
        make_request(dataset, path, out, session)


def make_request(dataset, path, out, session):
  """Extract embedding vectors for the given input path."""
  reader = tf.python_io.tf_record_iterator(
      path, options=tf.python_io.TFRecordOptions(
          compression_type=TFRecordCompressionType.GZIP))
  print >> sys.stderr, path
  for buf in reader:
    examples_tensor = json.loads(
        session.graph.get_collection('inputs')[0])['examples']
    output_tensor = json.loads(
        session.graph.get_collection('outputs')[0])[FLAGS.fetch]
    outputs = session.run(fetches=[output_tensor],
                          feed_dict={examples_tensor: [buf]})
    example = tf.train.Example.FromString(buf)
    sample_name = example.features.feature['sample_name'].bytes_list.value[0]
    super_population = example.features.feature[
        'super_population_string'].bytes_list.value[0]
    population = example.features.feature[
        'population_string'].bytes_list.value[0]
    print >> out, '%s\t%s\t%s\t%s\t%s' % (
        dataset, sample_name, super_population, population, '\t'.join(
            str(value) for value in outputs[0][0]))


def main(_):
  extract_embeddings()


if __name__ == '__main__':
  tf.app.run()
