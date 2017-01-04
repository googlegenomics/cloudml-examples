#!/bin/sh

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

# start_docker.sh
#
# Start a docker image with CloudML dependencies and mount a
# local directory. Usage:
#
#     ./start_docker.sh
#
# This script will ensure that the most recent version of the docker
# image is used.  It will also expose port 8080 to allow the use of
# tensorboard to visualize progress of model training.

docker pull gcr.io/cloud-datalab/datalab:local

docker run -it -p "127.0.0.1:8080:8080" \
       --entrypoint=/bin/bash \
       --volume $HOME/dockerVolume:/mnt/code \
       --workdir /mnt/code \
       gcr.io/cloud-datalab/datalab:local
