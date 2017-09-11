### Disclaimer

This is not an official Google product.

cloudml-examples
================

This repository contains an example of applying machine learning to genomic data using [Cloud Machine Learning Engine (Cloud ML Engine)](https://cloud.google.com/ml-engine/). The learning problem demonstrated is an ancestry inference. Identification of genetic ancestry is important for adjusting putative genetic associations with traits that may be driven largely through population structure. It is also important for assessing the accuracy of self-reported ancestry.

The instructions below train a model to predict 1000 Genomes super population labels. The training data are the [1000 Genomes](http://googlegenomics.readthedocs.io/en/latest/use_cases/discover_public_data/1000_genomes.html) phase 3 variants stored in [Google BigQuery](https://cloud.google.com/bigquery/). The validation data are the [Simons Genome Diversity Project](http://googlegenomics.readthedocs.io/en/latest/use_cases/discover_public_data/simons_foundation.html) variants stored in BigQuery. The training data is pre-processed using pipelines written with [Apache Beam](https://beam.apache.org/) and executed on [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs/).

This approach uses continuous vectors of genomic variants for analysis and inference on Machine Learning pipelines. For related work, see also [Diet Networks: Thin Parameters for Fat Genomics](https://openreview.net/pdf?id=Sk-oDY9ge) Romero et. al.

This is a non-trivial example in terms of cost (it may consume a large portion
of the [free trial credit](https://cloud.google.com/free/)) and also in terms of
the variety of tools used. We suggest working through the introductory materials
for each tool before working with the code in this repository.

## Getting Started

1. [Set up the Dataflow SDK for Python](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)

2. [Set up Cloud ML Engine](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)

3. This code depends on a few additional python packages. If you are
using [virtualenv](https://virtualenv.pypa.io/), the following commands will
create a virtualenv, activate it, and install those dependencies.

  ```bash
  virtualenv --system-site-packages ~/virtualEnvs/tensorflow
  source ~/virtualEnvs/tensorflow/bin/activate
  pip install --upgrade pip jinja2 pyfarmhash google-cloud-dataflow tensorflow
  ```
4. Set some environment variables to make copy/pasting commands a bit easier.

  * `PROJECT_ID=<YOUR_PROJECT>`
  * `BUCKET=gs://<YOUR_BUCKET>` this should be the **regional** bucket you
  created during Cloud ML Engine setup.

5. git clone this repository

## Pre-processing using Apache Beam

*   See if a query for the data you want to work with is already available in the [`preprocess`](./preprocess) directory. If not:
    *   See also [Select Genomic Data to work with](http://googlegenomics.readthedocs.io/en/latest/sections/select_genomic_data.html) for other public data and how to load your own data.
    *   Write jinja files containing the queries for your desired data.
*   Run a [`preprocess_data.py`](./trainer/preprocess_data.py) pipeline to convert
the data from BigQuery to TFRecords in Cloud Storage. For example:

Preprocess training data:

```bash
python -m trainer.preprocess_data \
  --setup_file ./setup.py \
  --output ${BUCKET}/1000-genomes \
  --project ${PROJECT_ID} \
  --metadata ./preprocess/1000_genomes_metadata.jinja \
  --input ./preprocess/1000_genomes_phase3_b37.jinja \
  --runner DataflowRunner \
  --worker_machine_type n1-highmem-8 \
  --no_hethom_words
```

Preprocess validation data:

```bash
python -m trainer.preprocess_data \
  --setup_file ./setup.py \
  --output ${BUCKET}/sgdp \
  --project ${PROJECT_ID} \
  --metadata ./preprocess/sgdp_metadata.jinja \
  --input ./preprocess/sgdp_data_b37.jinja \
  --runner DataflowRunner \
  --no_hethom_words
```

## Training using CloudML

```bash
EXAMPLES_SUBDIR=<the date-time subdirectory created during the training data preprocess step>
JOB_NAME=super_population_1000_genomes
gcloud ml-engine jobs submit training ${JOB_NAME} \
  --project ${PROJECT_ID} \
  --region us-central1 \
  --config config.yaml \
  --package-path ./trainer \
  --module-name trainer.variants_inference \
  --job-dir ${BUCKET}/models/${JOB_NAME} \
  --runtime-version 1.2 \
  -- \
  --input_dir ${BUCKET}/1000-genomes/${EXAMPLES_SUBDIR}/ \
  --sparse_features all_not_x_y \
  --num_classes 5 \
  --eval_labels="AFR,AMR,EAS,EUR,SAS" \
  --target_field super_population \
  --hidden_units 20 \
  --num_buckets 50000 \
  --num_train_steps 10000
```

If training results in an out of memory exception, add argument `--num_eval_steps 1` to the command line.

To inspect the behavior of training, launch TensorBoard and point it at the summary logs produced during training — both during and after execution.

```bash
tensorboard --port=8080 \
    --logdir ${BUCKET}/models/${JOB_NAME}/
```

*Tip: When running all of these commands from [Google Cloud Shell](https://cloud.google.com/shell/docs/), the [web preview](https://cloud.google.com/shell/docs/using-web-preview) feature can be used to view the TensorBoard user interface.*

The model generally converges sooner than 10,000 steps and you'll see this via TensorBoard. Training can be stopped early to avoid overfitting. To obtain the "saved model" needed for prediction, start training again from the exact same output directory (it will pick up where it left off) and have it run for a few more steps than it has already completed.

For example, if the job was cancelled after completing step 5,632, the following command will trigger a save model operation.

``` bash
gcloud ml-engine jobs submit training ${JOB_NAME}_save_model \
  ... <all other flags same as above>
  --num_train_steps 5700
```

## Hyperparameter tuning

Cloud ML Engine provides out of the box support for [Hyperparameter
tuning](https://cloud.google.com/ml-engine/docs/concepts/hyperparameter-tuning-overview). Running Hyperparameter tuning job is exactly same as a training job except you need to provide options in [TrainingInput](https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput).

```bash
EXAMPLES_SUBDIR=<the date-time subdirectory created during the validation data preprocess step>
gcloud ml-engine jobs submit training ${JOB_NAME} \
  --project ${PROJECT_ID} \
  --region us-central1 \
  --package-path ./trainer \
  --module-name trainer.variants_inference \
  --job-dir ${BUCKET}/hptuning/${JOB_NAME} \
  --config hptuning_config.yaml \
  -- \
  --input_dir ${BUCKET}/1000-genomes/${EXAMPLES_SUBDIR}/examples* \
  --sparse_features all_not_x_y \
  --num_classes 5 \
  --eval_labels="AFR,AMR,EAS,EUR,SAS" \
  --target_field super_population \
  --hidden_units 20 \
  --num_buckets 50000 \
  --num_train_steps 10000
```

## Batch predict

```bash
EXAMPLES_SUBDIR=<the date-time subdirectory created during the validation data preprocess step>
EXPORT_SUBDIR=<model subdirectory underneath 'export/Servo/'>
gcloud --project ${PROJECT_ID} ml-engine jobs submit \
    prediction ${JOB_NAME}_predict \
    --model-dir \
        ${BUCKET}/models/${JOB_NAME}/export/Servo/${EXPORT_SUBDIR} \
    --input-paths ${BUCKET}/sgdp/${EXAMPLES_SUBDIR}/examples* \
    --output-path ${BUCKET}/predictions/${JOB_NAME} \
    --region us-central1 \
    --data-format TF_RECORD_GZIP
```

If prediction yields an error regarding the size of the saved model, request more quota for your project.
