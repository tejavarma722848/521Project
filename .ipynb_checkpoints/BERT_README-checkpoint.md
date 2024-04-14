# BERT Text Classification README

## Introduction

This repository contains code for text classification using BERT (Bidirectional Encoder Representations from Transformers), a powerful pre-trained language representation model.

## Files Included

- `train_test_split.py`: Script to split the dataset into training and testing sets.
- `data_preprocessing.py`: Script to preprocess the dataset by balancing the number of positive and negative reviews and converting labels to one-hot encoding.
- `bert_classification.py`: Script implementing the BERT model for text classification.
  
## Setup

Before running the scripts, ensure you have the following dependencies installed:

- tensorflow
- tensorflow_hub
- scikit-learn
- pandas
- numpy

Additionally, make sure to download the BERT model from the TensorFlow Hub and set the appropriate paths in the scripts.

## Usage

### train_test_split.py:

- Execute this script to split your dataset into training and testing sets.
- Set the path to your CSV file containing the dataset in the `file_path` variable.
- Adjust the sample sizes for training and testing datasets as needed.

### data_preprocessing.py:

- Run this script after splitting the dataset to balance the number of positive and negative reviews.
- Set the path to your CSV file in the `file_path` variable.
- The script will create a new column 'Sentiment' based on the 'Score' column, mapping 0 and 1 ratings to negative and positive sentiment, respectively.

### bert_classification.py:

- This script implements the BERT model for text classification.
- Before running, ensure you have created BERT-compatible features using the `convert_examples_to_features` function.
- Define the model architecture in the `create_model` function.
- Train the model by specifying the necessary parameters such as learning rate, batch size, and number of epochs.
- Evaluate the model on the test set and calculate performance metrics such as accuracy, precision, recall, and F1-score.

## `model_fn_builder` Function

The `model_fn_builder` function creates our model function using the passed parameters for `num_labels`, `learning_rate`, `num_train_steps`, and `num_warmup_steps`. It returns a `model_fn` closure for `TPUEstimator`.

## Training Process

- Batch Size: 32
- Learning Rate: 2e-5
- Number of Training Epochs: 4.0
- Warmup Proportion: 0.1
- Save Checkpoints Steps: 500
- Save Summary Steps: 100

The training process involves creating an Estimator using the defined model function, run configuration, and additional parameters. The model is trained using the input function for training data, and evaluation is performed on the test data using the input function for testing.

## References

- BERT: [GitHub Repository](https://github.com/google-research/bert)
- TensorFlow Hub: [Official Documentation](https://www.tensorflow.org/hub)
- Transformers library: [Hugging Face](https://huggingface.co/transformers/)
