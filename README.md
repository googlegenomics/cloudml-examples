### Disclaimer

This is not an official Google product.

cloudml-examples
================

This repository contains an example of applying machine learning to genomic data using [Cloud Machine Learning Engine (Cloud ML Engine)](https://cloud.google.com/ml-engine/). The learning problem demonstrated is an ancestry inference. Identification of genetic ancestry is important for adjusting putative genetic associations with traits that may be driven largely through population structure. It is also important for assessing the accuracy of self-reported ancestry.

The instructions below train a model to predict 1000 Genomes super population labels. The training data are the [1000 Genomes](https://cloud.google.com/genomics/docs/public-datasets/1000-genomes) phase 3 variants stored in [Google BigQuery](https://cloud.google.com/bigquery/). The validation data are the [Simons Genome Diversity Project](https://cloud.google.com/genomics/docs/public-datasets/simons) variants stored in BigQuery. The training data is pre-processed using pipelines written with [Apache Beam](https://beam.apache.org/) and executed on [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs/).

This approach uses continuous vectors of genomic variants for analysis and inference on Machine Learning pipelines. For related work, see also [Diet Networks: Thin Parameters for Fat Genomics](https://openreview.net/pdf?id=Sk-oDY9ge) Romero et. al.

This is a non-trivial example in terms of cost (it may consume a large portion
of the [free trial credit](https://cloud.google.com/free/)) and also in terms of
the variety of tools used. We suggest working through the introductory materials
for each tool before working with the code in this repository.

## Blog Post

[Genomic ancestry inference with deep learning](https://cloud.google.com/blog/big-data/2017/09/genomic-ancestry-inference-with-deep-learning) blog post provides a great overview of the end-to-end reference implementation. It also links to pre-processed data and trained model in Google Cloud Storage if you would like to skip some of the steps below.

## Getting Started

1. [Set up the Dataflow SDK for Python](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)

2. [Set up Cloud ML Engine](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)

3. This code depends on a few additional python packages. If you are
using [virtualenv](https://virtualenv.pypa.io/), the following commands will
create a virtualenv, activate it, and install those dependencies.

    ```
    virtualenv --system-site-packages ~/virtualEnvs/tensorflow
    source ~/virtualEnvs/tensorflow/bin/activate
    pip2.7 install --upgrade pip jinja2 pyfarmhash apache-beam[gcp] tensorflow
    ```

4. Set some environment variables to make copy/pasting commands a bit easier.

  * `PROJECT_ID=<YOUR_PROJECT>`
  * `BUCKET=gs://<YOUR_BUCKET>` this should be the **regional** bucket you
  created during Cloud ML Engine setup.

5. git clone this repository and change into its directory

## Pre-processing using Apache Beam

*   See if a query for the data you want to work with is already available in the [`preprocess`](./preprocess) directory. If not:
    *   See also [Select Genomic Data to work with](https://cloud.google.com/genomics/docs/public-datasets/) for other public data and how to load your own data.
    *   Write jinja files containing the queries for your desired data.
*   Run a [`preprocess_data.py`](./trainer/preprocess_data.py) pipeline to convert
the data from BigQuery to TFRecords in Cloud Storage. For example:

Preprocess training data:

```
python2.7 -m trainer.preprocess_data \
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

```
python2.7 -m trainer.preprocess_data \
  --setup_file ./setup.py \
  --output ${BUCKET}/sgdp \
  --project ${PROJECT_ID} \
  --metadata ./preprocess/sgdp_metadata.jinja \
  --input ./preprocess/sgdp_data_b37.jinja \
  --runner DataflowRunner \
  --no_hethom_words
```

## Training using CloudML

```
EXAMPLES_SUBDIR=<the date-time subdirectory created during the training data preprocess step>
JOB_NAME=super_population_1000_genomes
gcloud ai-platform jobs submit training ${JOB_NAME} \
  --project ${PROJECT_ID} \
  --region us-central1 \
  --config config.yaml \
  --package-path ./trainer \
  --module-name trainer.variants_inference \
  --job-dir ${BUCKET}/models/${JOB_NAME} \
  --runtime-version 1.2 \
  -- \
  --input_dir ${BUCKET}/1000-genomes/${EXAMPLES_SUBDIR}/ \
  --export_dir ${BUCKET}/models/${JOB_NAME} \
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

```
tensorboard --port=8080 \
    --logdir ${BUCKET}/models/${JOB_NAME}/
```

*Tip: When running all of these commands from [Google Cloud Shell](https://cloud.google.com/shell/docs/), the [web preview](https://cloud.google.com/shell/docs/using-web-preview) feature can be used to view the TensorBoard user interface.*

The model generally converges sooner than 10,000 steps and you'll see this via TensorBoard. Training can be stopped early to avoid overfitting. To obtain the "saved model" needed for prediction, start training again from the exact same output directory (it will pick up where it left off) and have it run for a few more steps than it has already completed.

For example, if the job was cancelled after completing step 5,632, the following command will trigger a save model operation.

```
gcloud ai-platform jobs submit training ${JOB_NAME}_save_model \
  ... <all other flags same as above>
  --num_train_steps 5700
```

## Hyperparameter tuning

Cloud ML Engine provides out of the box support for [Hyperparameter
tuning](https://cloud.google.com/ml-engine/docs/concepts/hyperparameter-tuning-overview). Running Hyperparameter tuning job is exactly same as a training job except you need to provide options in [TrainingInput](https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#traininginput).

```
EXAMPLES_SUBDIR=<the date-time subdirectory created during the validation data preprocess step>
gcloud ai-platform jobs submit training ${JOB_NAME} \
  --project ${PROJECT_ID} \
  --region us-central1 \
  --package-path ./trainer \
  --module-name trainer.variants_inference \
  --job-dir ${BUCKET}/hptuning/${JOB_NAME} \
  --config hptuning_config.yaml \
  -- \
  --input_dir ${BUCKET}/1000-genomes/${EXAMPLES_SUBDIR}/examples* \
  --export_dir ${BUCKET}/hptuning/${JOB_NAME} \
  --sparse_features all_not_x_y \
  --num_classes 5 \
  --eval_labels="AFR,AMR,EAS,EUR,SAS" \
  --target_field super_population \
  --hidden_units 20 \
  --num_buckets 50000 \
  --num_train_steps 10000
```

## Batch predict

```
EXAMPLES_SUBDIR=<the date-time subdirectory created during the validation data preprocess step>
EXPORT_SUBDIR=<model subdirectory underneath 'export/Servo/'>
gcloud --project ${PROJECT_ID} ai-platform jobs submit \
    prediction ${JOB_NAME}_predict \
    --model-dir \
        ${BUCKET}/models/${JOB_NAME}/export/Servo/${EXPORT_SUBDIR} \
    --input-paths ${BUCKET}/sgdp/${EXAMPLES_SUBDIR}/examples* \
    --output-path ${BUCKET}/predictions/${JOB_NAME} \
    --region us-central1 \
    --data-format TF_RECORD_GZIP
```

If prediction yields an error regarding the size of the saved model, request more quota for your project.

## Examine the prediction results

For Simons Genome Diversity project data, one might examine the prediction results as follows:

```
bq load --source_format NEWLINE_DELIMITED_JSON --autodetect \
  YOUR-DATASET.sgdp_ancestry_prediction_results \
  ${BUCKET}/predictions/prediction.results*
```

```
SELECT
  key,
  probabilities[ORDINAL(1)] AS AFR,
  probabilities[ORDINAL(2)] AS AMR,
  probabilities[ORDINAL(3)] AS EAS,
  probabilities[ORDINAL(4)] AS EUR,
  probabilities[ORDINAL(5)] AS SAS,
  info.*
FROM
  `YOUR-DATASET.sgdp_ancestry_prediction_results`
JOIN
  `bigquery-public-data.human_genome_variants.simons_genome_diversity_project_sample_attributes` AS info
ON
  key = id_from_vcf
ORDER BY
  region, population
```

If you are using the BigQuery web UI, you can click on `Save to GoogleSheets` and then in GoogleSheets:

* select the 5 columns of prediction probabilities
* `Format` -> `Conditional Formatting` -> `Color Scale` and use bright yellow for `Max Value`
