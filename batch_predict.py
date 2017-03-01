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
"""Batch prediction pipeline for variants inference."""

import argparse
import datetime
import os
import uuid

import apache_beam as beam

import google.cloud.ml as ml


parser = argparse.ArgumentParser(
    description='Runs batch prediction.')
parser.add_argument(
    '--project_id',
    required=True,
    help='The project to which the job will be submitted.')
parser.add_argument(
    '--runner',
    default='DirectRunner',
    help='The beam runner to use.')
parser.add_argument(
    '--model_dir',
    help='The model from which to predict results.')
parser.add_argument(
    '--input',
    required=True,
    help='The file glob for TFRecordIO Example protos as input '
    'to the prediction.')
parser.add_argument(
    '--deploy_model_name',
    default='variants_inference',
    help=('If --cloud is used, the model is deployed with this '
          'name. The default is variants_inference.'))
parser.add_argument(
    '--deploy_model_version',
    default='v' + uuid.uuid4().hex[:4],
    help=('If --cloud is used, the model is deployed with this '
          'version. The default is four random characters.'))
parser.add_argument(
    '--output',
    required=True,
    help='Output directory to which to write prediction results.')

args, passthrough_args = parser.parse_known_args()

args.trainer_job_args = passthrough_args


def deploy_model(model_name, version_name, trained_model):
  return trained_model | ml.DeployVersion(model_name, version_name)


def dataflow():
  """Run the Dataflow pipeline."""
  options = {
      'staging_location':
          os.path.join(args.output, 'tmp', 'staging'),
      'temp_location':
          os.path.join(args.output, 'tmp'),
      'job_name': ('batch-predict-' +
                   datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
      'project':
          args.project_id,
      # Specify the version of the ml.sdk_location to use plus the path to the
      # training code on GCS.
      'extra_packages': [ml.sdk_location],
      # THROUGHPUT_BASED is not the default value and might generate many
      # small files depending on the number of workers.
      'autoscaling_algorithm':
          'THROUGHPUT_BASED'
  }
  opts = beam.pipeline.PipelineOptions(flags=[], **options)
  p = beam.Pipeline(args.runner, options=opts)

  if args.model_dir:
    trained_model = (p
                     | beam.Create([args.model_dir]))
    deployed = deploy_model(args.deploy_model_name,
                            args.deploy_model_version, trained_model)
  else:
    deployed = (p
                | beam.Create([[(args.deploy_model_name,
                                 args.deploy_model_version)]]))

  # Use deployed model to run a batch prediction.
  output_uri = os.path.join(args.output, 'batch_prediction_results')
  _ = deployed | 'Batch Predict' >> ml.Predict(
      [args.input],
      output_uri,
      region='us-central1',
      data_format='TF_RECORD_GZIP')

  p.run().wait_until_finish()


def main():
  dataflow()


if __name__ == '__main__':
  main()
