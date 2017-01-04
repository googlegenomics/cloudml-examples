#!/bin/bash

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup_docker.sh
#
# This script will add/update additional CloudML dependencies inside
# the running docker image.
#
# Place this script in the local directory to be mounted by the container
# and run it within the Docker image.  Usage:
#
#     ./setup_docker.sh YOUR-PROJECT-ID

set -o errexit
set -o nounset

readonly PROJECT_ID="${1}"

# Needed for subsequent installation of libjpeg-dev in the next step.
apt-get update

# Install required tools and dependencies per
# https://cloud.google.com/ml/docs/how-tos/getting-set-up
curl https://raw.githubusercontent.com/GoogleCloudPlatform/cloudml-samples/master/tools/setup_docker.sh | bash

# The hash algorithm used to convert variants represented
# as "words" into integers.
apt-get install -y build-essential
pip install pyfarmhash

# Needed by Dataflow.
gcloud auth login

# Needed by CloudML.
gcloud auth application-default login

# Needed by environment verification.
gcloud config set project "${PROJECT_ID}"

# Verify your environment per
# https://cloud.google.com/ml/docs/how-tos/getting-set-up
curl https://raw.githubusercontent.com/GoogleCloudPlatform/cloudml-samples/master/tools/check_environment.py | python

