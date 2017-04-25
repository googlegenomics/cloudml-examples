### Disclaimer

This is not an official Google product.

cloudml-examples
================

This repository contains an example of applying machine learning to genomic data using Google Cloud ML. The particular learning problem demonstrated is an ancestry inference. Identification of genetic ancestry is important for adjusting putative genetic associations with traits that may be driven largely through population structure. It is also important for assessing the accuracy of self-reported ancestry.

The instructions below train a model to predict 1000 Genomes super population labels.  The training data are the [1000 Genomes](http://googlegenomics.readthedocs.io/en/latest/use_cases/discover_public_data/1000_genomes.html) phase 3 variants stored in BigQuery. The evaluation data are the [Simons Genome Diversity Project](http://googlegenomics.readthedocs.io/en/latest/use_cases/discover_public_data/simons_foundation.html) variants stored in BigQuery.  The training data is pre-processed using pipelines written with [Apache Beam](https://beam.apache.org/) and executed on [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs/). Prediction also uses Beam and Dataflow and the results can be written to the local file system or Cloud Storage.

This approach uses continuous vectors of genomic variants for analysis and inference on Machine Learning pipelines. For related work, see also [Diet Networks: Thin Parameters for Fat Genomics](https://openreview.net/pdf?id=Sk-oDY9ge) Romero et. al.

## Getting Started

The [CloudML setup instructions](https://cloud.google.com/ml/docs/how-tos/getting-set-up) list a few ways to get setup. We recommend the Docker setup option and have included some scripts to make that a bit easier.

*   One time setup:
    *   [Set up CloudML](https://cloud.google.com/ml/docs/how-tos/getting-set-up) by following the Docker Container instructions.
    *   Create a directory which will be mounted inside of your Docker container: `mkdir $HOME/dockerVolume`
    *   Copy this tree of code into `$HOME/dockerVolume`

*   To update your environment to the latest code:
    *   Run [`start_docker.sh`](./docker/start_docker.sh) from your shell to start the container.
    *   Run [`setup_docker.sh`](./docker/setup_docker.sh) *within the container* to configure it further.
    *   Set some environment variants to make copy/pasting commands a bit easier.
        * `PROJECT_ID=<YOUR_PROJECT>`
        * `BUCKET_NAME=<YOUR_BUCKET>` this should be the **regional** bucket you created during CloudML setup.

## Pre-processing using Apache Beam

*   See if a query for the data you want to work with is already available in the [`preprocess`](./preprocess) directory.  If not:
    *   See also [Select Genomic Data to work with](http://googlegenomics.readthedocs.io/en/latest/sections/select_genomic_data.html) for other public data and how to load your own data.
    *   Write jinja files containing the queries for your desired data.
    *   Copy the jinja files into `$HOME/dockerVolume`
*   *From within the Docker container* run pipeline
   [`preprocess_data.py`](./trainer/preprocess_data.py) to convert the data from BigQuery
    to TFRecords in Cloud Storage. For example:

Preprocess training data:

```bash
python -m trainer.preprocess_data \
  --setup_file ./setup.py \
  --output gs://${BUCKET_NAME}/1000-genomes \
  --project ${PROJECT_ID} \
  --metadata ./preprocess/1000_genomes_metadata.jinja \
  --input ./preprocess/1000_genomes_phase3_b37.jinja \
  --runner DataflowRunner \
  --worker_machine_type n1-highmem-8 \
  --no_hethom_words
```

Preprocess evaluation data:

```bash
python -m trainer.preprocess_data \
  --setup_file ./setup.py \
  --output gs://${BUCKET_NAME}/sgdp \
  --project ${PROJECT_ID} \
  --metadata ./preprocess/sgdp_metadata.jinja \
  --input ./preprocess/sgdp_data_b37.jinja \
  --runner DataflowRunner \
  --no_hethom_words
```

## Training using CloudML

```bash
EXAMPLES_SUBDIR=<the date-time subdirectory created during the training data preprocess step>
gcloud ml-engine jobs submit training MY_JOB_NAME \
  --config config.yaml \
  --region us-central1 \
  --module-name trainer.variants_inference \
  --staging-bucket gs://${BUCKET_NAME} \
  --package-path ./trainer \
  --project ${PROJECT_ID} \
  -- \
  --input_dir gs://${BUCKET_NAME}/1000-genomes/${EXAMPLES_SUBDIR}/ \
  --sparse_features all_not_x_y \
  --num_classes 5 \
  --eval_labels="AFR,AMR,EAS,EUR,SAS" \
  --target_field super_population \
  --hidden_units 20 \
  --output_path gs://${BUCKET_NAME}/models/1000-genomes-super-population/ \
  --num_train_steps 50000 \
  --num_buckets 50000
```

If training results in an out of memory exception, add argument `--num_eval_steps 1` to the command line.

To inspect the behavior of training, launch TensorBoard and point it at the summary logs produced during training — both during and after execution.  *From within the Docker container*, run the following command:

```bash
tensorboard --port=8080 \
    --logdir gs://${BUCKET_NAME}/models/1000-genomes-super-population/
```

*Tip: When running all of these commands from [Google Cloud Shell](https://cloud.google.com/shell/docs/), the [web preview](https://cloud.google.com/shell/docs/using-web-preview) feature can be used to view the tensorboard user interface.*

## Batch predict

```bash
EXAMPLES_SUBDIR=<the date-time subdirectory created during the evaluation data preprocess step>
EXPORT_SUBDIR=<model subdirectory underneath 'export/Servo/'>
gcloud --project ${PROJECT_ID} ml-engine jobs submit \
    prediction ${JOB_NAME}_predict \
    --model-dir \
        gs://${BUCKET_NAME}/models/${JOB_NAME}/export/Servo/${EXPORT_SUBDIR} \
    --input-paths gs://${BUCKET_NAME}/sgdp/${EXAMPLES_SUBDIR}/examples* \
    --output-path gs://${BUCKET_NAME}/predictions/${JOB_NAME} \
    --region us-central1 \
    --data-format TF_RECORD_GZIP
```

If prediction yields an error regarding the size of the saved model, request more quota for your project.
