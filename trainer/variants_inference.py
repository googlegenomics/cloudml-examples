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
"""Tensorflow implementation of variants inference."""

import functools
import json
import os


import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType


EXAMPLE_KEY = "input_feature"
CONCAT_EMBEDDINGS_KEY = "concat_embeddings"

flags = tf.flags
logging = tf.logging
metrics_lib = tf.contrib.metrics

flags.DEFINE_float(
    "learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float(
    "momentum", 0.9, "Momentum.")
flags.DEFINE_integer(
    "num_classes", None, "The number of classes on the dataset.")
flags.DEFINE_integer(
    "hidden_units", None, "The number of hidden units on the hidden layer.")
flags.DEFINE_integer(
    "num_buckets", 100000, "The number of buckets to use for hashing.")
flags.DEFINE_integer(
    "embedding_dimension", 100,
    "The total embedding dimension is obtained by multiplying this number "
    "by the number of feature columns.")
flags.DEFINE_integer(
    "batch_size", 50, "The size of the train and test batches.")
flags.DEFINE_integer(
    "feature_queue_capacity", 5, "The size of the feature queue.")
flags.DEFINE_string(
    "input_dir", None, "Path to the input files.")
flags.DEFINE_string(
    "eval_dir", None,
    "If specified use a separate eval dataset. The dataset labels and label "
    "indexes should match exactly those of the training set for the evaluation "
    "to work correctly. In other words you only need to use this if you want a "
    " different train/eval split than the one provided by default.")
flags.DEFINE_string(
    "output_path", "",
    "Base output directory. Used by the local and cloud jobs.")
flags.DEFINE_boolean(
    "use_integerized_features", True,
    "Whether the features are int64 values.")
flags.DEFINE_boolean(
    "use_gzip", True,
    "Whether the tfrecord files are compressed.")
tf.flags.DEFINE_integer(
    "num_train_steps", 10000,
    "Number of training iterations. None means continuous training.")
tf.flags.DEFINE_integer(
    "num_eval_steps", 10,
    "Number of evaluation iterations. When running continuous_eval, this is "
    "the number of eval steps run for each evaluation of a checkpoint.")
flags.DEFINE_string(
    "target_field", None,
    "The name of the field that contains the labels.")
flags.DEFINE_string(
    "id_field", "sample_name",
    "The name of the field that contains the sample ids.")
flags.DEFINE_string(
    "sparse_features", None,
    "A list of the sparse features to process. For example "
    "variants_2,variants_17.  Alternatively specify 'all_not_x_y' "
    "to indicate chromosomes 1 through 22.")
flags.DEFINE_string(
    "eval_labels", "",
    "Optional, comma separated values of the labels used for"
    "per-class evaluation, the order should be the same as the one used "
    "when extracting the features.")

FLAGS = flags.FLAGS

SPARSE_FEATURE_NAMES = ["variants"]


def _get_feature_names():
  sparse_features = FLAGS.sparse_features.split(",")
  if sparse_features and sparse_features[0] == "all_not_x_y":
    return ["variants_%s" % ref for ref in range(1, 23)]
  else:
    return sparse_features


def _get_eval_labels():
  return enumerate(FLAGS.eval_labels.split(",")) if FLAGS.eval_labels else []


def _get_feature_columns(include_target_column):
  """Generates a tuple of `FeatureColumn` objects for our inputs.

  Args:
    include_target_column: Whether to include the target columns.

  Returns:
    Tuple of `FeatureColumn` objects.
  """
  embedding_columns = []
  for column_name in _get_feature_names():
    if FLAGS.use_integerized_features:
      sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
          column_name=column_name,
          bucket_size=FLAGS.num_buckets,
          combiner="sqrtn",
          dtype=tf.int64)
    else:
      sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
          column_name=column_name,
          hash_bucket_size=FLAGS.num_buckets,
          combiner="sqrtn",
          dtype=tf.string)

    embedding = tf.contrib.layers.embedding_column(
        sparse_id_column=sparse_column,
        combiner="sqrtn",
        dimension=FLAGS.embedding_dimension)
    embedding_columns.append(embedding)
  feature_columns = tuple(sorted(embedding_columns))
  if include_target_column:
    label_column = tf.contrib.layers.real_valued_column(
        FLAGS.target_field, dtype=tf.int64)
    feature_columns += (label_column,)
  return feature_columns


def _build_input_fn(input_file_pattern, batch_size, mode):
  """Build input function.

  Args:
    input_file_pattern: The file patter for examples
    batch_size: Batch size
    mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

  Returns:
    Tuple, dictionary of feature column name to tensor and labels.
  """
  def _input_fn():
    """Supplies the input to the model.

    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).
    """
    logging.info("Reading files from %s", input_file_pattern)
    input_files = sorted(list(tf.gfile.Glob(input_file_pattern)))
    logging.info("Reading files from %s", input_files)
    include_target_column = (mode != tf.contrib.learn.ModeKeys.INFER)
    features_spec = tf.contrib.layers.create_feature_spec_for_parsing(
        feature_columns=_get_feature_columns(include_target_column))

    if FLAGS.use_gzip:
      def gzip_reader():
        return tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(
                compression_type=TFRecordCompressionType.GZIP))
      reader_fn = gzip_reader
    else:
      reader_fn = tf.TFRecordReader

    features = tf.contrib.learn.io.read_batch_features(
        file_pattern=input_files,
        batch_size=batch_size,
        queue_capacity=3*batch_size,
        randomize_input=mode == tf.contrib.learn.ModeKeys.TRAIN,
        feature_queue_capacity=FLAGS.feature_queue_capacity,
        reader=reader_fn,
        features=features_spec)
    target = None
    if include_target_column:
      target = features.pop(FLAGS.target_field)
    return features, target

  return _input_fn


def _predict_input_fn():
  """Supplies the input to the model.

  Returns:
    A tuple consisting of 1) a dictionary of tensors whose keys are
    the feature names, and 2) a tensor of target labels if the mode
    is not INFER (and None, otherwise).
  """
  feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
      feature_columns=_get_feature_columns(include_target_column=False))

  feature_spec[FLAGS.id_field] = tf.FixedLenFeature([], dtype=tf.string)
  feature_spec[FLAGS.target_field + "_string"] = tf.FixedLenFeature(
      [], dtype=tf.string)

  # Add a placeholder for the serialized tf.Example proto input.
  examples = tf.placeholder(tf.string, shape=(None,), name="examples")

  features = tf.parse_example(examples, feature_spec)
  # Pass the input tensor so it can be used for export.
  features[EXAMPLE_KEY] = examples
  return features, None


def _build_model_fn():
  """Build model function.

  Returns:
    A model function that can be passed to `Estimator` constructor.
  """
  def _model_fn(features, labels, mode):
    """Creates the prediction and its loss.

    Args:
      features: A dictionary of tensors keyed by the feature name.
      labels: A tensor representing the labels.
      mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.

    Returns:
      A tuple consisting of the prediction, loss, and train_op.
    """
    # Generate one embedding per sparse feature column and concatenate them.
    concat_embeddings = tf.contrib.layers.input_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=_get_feature_columns(include_target_column=False))

    tf.add_to_collection(CONCAT_EMBEDDINGS_KEY, concat_embeddings)

    # Add one hidden layer.
    hidden_layer_0 = tf.contrib.layers.relu(
        concat_embeddings, FLAGS.hidden_units)

    # Output and logistic loss.
    logits = tf.contrib.layers.linear(hidden_layer_0, FLAGS.num_classes)

    predictions = tf.contrib.layers.softmax(logits)
    if mode == tf.contrib.learn.ModeKeys.INFER:
      return predictions, None, None

    target_one_hot = tf.one_hot(labels, FLAGS.num_classes)
    target_one_hot = tf.reduce_sum(
        input_tensor=target_one_hot, reduction_indices=[1])
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target_one_hot)
    if mode == tf.contrib.learn.ModeKeys.EVAL:
      return predictions, loss, None

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=FLAGS.learning_rate,
        optimizer=opt)
    return predictions, loss, train_op

  return _model_fn


def _create_evaluation_metrics():
  """Creates the evaluation metrics for the model.

  Returns:
    A dictionary with keys that are strings naming the evaluation
      metrics and values that are functions taking arguments of
      (predictions, targets), returning a tuple of a tensor of the metric's
      value together with an op to update the metric's value.
  """
  eval_metrics = {}
  for k in [1]:
    eval_metrics["precision_at_%d" % k] = functools.partial(
        tf.contrib.metrics.streaming_sparse_precision_at_k, k=k)
    eval_metrics["recall_at_%d" % k] = functools.partial(
        tf.contrib.metrics.streaming_sparse_recall_at_k, k=k)

  for class_id, class_label in _get_eval_labels():
    k = 1
    eval_metrics["precision_at_%d_%s" % (k, class_label)] = functools.partial(
        tf.contrib.metrics.streaming_sparse_precision_at_k,
        k=k, class_id=class_id)
    eval_metrics["recall_at_%d_%s" % (k, class_label)] = functools.partial(
        tf.contrib.metrics.streaming_sparse_recall_at_k,
        k=k, class_id=class_id)
  return eval_metrics


def _signature_fn(examples, features, predictions):
  """Create a classification signature function and add to collections."""
  # Mark the inputs.
  inputs = {"examples": examples.name}
  tf.add_to_collection("inputs", json.dumps(inputs))

  concat_embeddings = tf.get_collection(CONCAT_EMBEDDINGS_KEY)[0]
  outputs = {"score": predictions.name,
             "key": features[FLAGS.id_field].name,
             "target": features[FLAGS.target_field + "_string"].name,
             "embeddings": concat_embeddings.name}
  tf.add_to_collection("outputs", json.dumps(outputs))

  output_signature = manifest_pb2.Signature()
  for name, tensor_name in outputs.iteritems():
    output_signature.generic_signature.map[name].tensor_name = tensor_name

  input_signature = manifest_pb2.Signature()
  for name, tensor_name in inputs.iteritems():
    input_signature.generic_signature.map[name].tensor_name = tensor_name

  # Create a classification signature for serving prediction.
  signature = manifest_pb2.Signature()
  signature.classification_signature.input.tensor_name = examples.name
  signature.classification_signature.scores.tensor_name = predictions.name

  # Returns a tuple of None default signature and a dictionary of named
  # signatures with inputs and outputs.
  return signature, {"inputs": input_signature, "outputs": output_signature}


def _get_export_monitor(output_dir):
  """Create an export monitor."""
  return tf.contrib.learn.monitors.ExportMonitor(
      input_fn=_predict_input_fn,
      input_feature_key=EXAMPLE_KEY,
      every_n_steps=2000,
      export_dir=os.path.join(output_dir, "export"),
      signature_fn=_signature_fn)


def _def_experiment(
    train_file_pattern, eval_file_pattern, batch_size):
  """Creates the function used to configure the experiment runner.

  This function creates a function that is used by the learn_runner
  module to create an Experiment.

  Args:
    train_file_pattern: The directory the train data can be found in.
    eval_file_pattern: The directory the test data can be found in.
    batch_size: Batch size

  Returns:
    A function that creates an Experiment object for the runner.
  """

  def _experiment_fn(output_dir):
    """Experiment function used by learn_runner to run training/eval/etc.

    Args:
      output_dir: String path of directory to use for outputs.

    Returns:
      tf.learn `Experiment`.
    """
    estimator = tf.contrib.learn.Estimator(
        model_fn=_build_model_fn(),
        model_dir=output_dir)
    train_input_fn = _build_input_fn(
        input_file_pattern=train_file_pattern,
        batch_size=batch_size,
        mode=tf.contrib.learn.ModeKeys.TRAIN)
    eval_input_fn = _build_input_fn(
        input_file_pattern=eval_file_pattern,
        batch_size=batch_size,
        mode=tf.contrib.learn.ModeKeys.EVAL)

    export_monitor = _get_export_monitor(output_dir)
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        train_steps=FLAGS.num_train_steps,
        train_monitors=[export_monitor],
        eval_input_fn=eval_input_fn,
        eval_steps=FLAGS.num_eval_steps,
        eval_metrics=_create_evaluation_metrics(),
        min_eval_frequency=100)

  return _experiment_fn


def main(unused_argv):
  if not FLAGS.input_dir:
    raise ValueError("Input dir should be specified.")

  if FLAGS.eval_dir:
    train_file_pattern = os.path.join(FLAGS.input_dir, "examples*")
    eval_file_pattern = os.path.join(FLAGS.eval_dir, "examples*")
  else:
    train_file_pattern = os.path.join(FLAGS.input_dir, "examples*[0-7]-of-*")
    eval_file_pattern = os.path.join(FLAGS.input_dir, "examples*[89]-of-*")

  if not FLAGS.num_classes:
    raise ValueError("Number of classes should be specified.")

  if not FLAGS.sparse_features:
    raise ValueError("Name of the sparse features should be specified.")

  learn_runner.run(
      experiment_fn=_def_experiment(
          train_file_pattern,
          eval_file_pattern,
          FLAGS.batch_size),
      output_dir=FLAGS.output_path)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
