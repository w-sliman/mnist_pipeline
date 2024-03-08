# TFX Pipeline for MNIST Classification

This TFX pipeline is designed to preprocess and train a model for classifying the MNIST dataset. It includes components for data ingestion, preprocessing, training, evaluation, and model deployment.

## Overview

The TFX pipeline automates the machine learning workflow, from data ingestion to model deployment. It leverages TensorFlow Extended (TFX) components to create a scalable and reproducible pipeline.

### Pipeline Components

- **ImportExampleGen**: Imports data from the specified location.
- **StatisticsGen**: Generates statistics from the dataset.
- **SchemaGen**: Generates a schema based on the dataset statistics.
- **ExampleValidator**: Validates examples based on the schema.
- **Transform**: Applies preprocessing to the dataset.
- **Trainer**: Trains a model using the preprocessed data.
- **Resolver**: Resolves the latest blessed model.
- **Evaluator**: Evaluates the trained model against a baseline.
- **Pusher**: Pushes the trained model to the serving directory.

## Installation

1. Clone the repository.
2. Create a virtual environment, I used venv.
3. Inside the environment, install tfx using pip.
4. Download the MNIST dataset Using the download_mnist.py script in data folder.
5. Create and Run the pipeline using the TFX CLI:
```bash
tfx pipeline update --pipeline_path local_runner.py --engine local
tfx run create --pipeline_name mnist_pipeline --engine local
```
4. Explore the output director and the metadata sqlite store.
