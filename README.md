# Sentiment Analysis

## Introduction

This project aims to perform sentiment analysis on Twitter data using various deep learning models. The dataset used for this analysis is Twitter_Data.

## Data Preprocessing

- The dataset is first loaded into a pandas DataFrame.
- The 'Score' column in the dataset is converted into sentiment labels, where 0 represents negative sentiment and 1 represents positive sentiment.
- To balance the dataset, an equal number of negative and positive samples are selected randomly.
- Text preprocessing steps include:
  - Converting text to lowercase.
  - Removing HTML tags, punctuation, and special characters.
  - Tokenization of sentences into words.
  - Removal of stopwords and lemmatization of words.

## Model Building

### LSTM Model

- A Long Short-Term Memory (LSTM) model is built using Keras.
- The model architecture includes an Embedding layer followed by a CuDNNLSTM layer and a Dense output layer with softmax activation.
- The model is compiled with categorical cross-entropy loss and Adam optimizer.

### CNN Model

- A Convolutional Neural Network (CNN) model is built using Keras.
- The model architecture includes an Embedding layer, Conv1D layer, MaxPooling1D layer, Dropout layer, Flatten layer, and a Dense output layer with softmax activation.
- The model is compiled with categorical cross-entropy loss and Adam optimizer.

### CNN-LSTM Model

- A hybrid CNN-LSTM model is built using Keras.
- The model architecture includes an Embedding layer, Conv1D layer, MaxPooling1D layer, Dropout layer, CuDNNLSTM layer, and a Dense output layer with softmax activation.
- The model is compiled with categorical cross-entropy loss and Adam optimizer.

### LSTM-CNN Model

- Another hybrid LSTM-CNN model is built using Keras.
- The model architecture includes an Embedding layer, CuDNNLSTM layer, Conv1D layer, MaxPooling1D layer, Dropout layer, Flatten layer, and a Dense output layer with softmax activation.
- The model is compiled with categorical cross-entropy loss and Adam optimizer.

## Model Training and Evaluation

- The models are trained on the balanced dataset with train-test split and cross-validation.
- Metrics such as accuracy, precision, recall, and F1 score are computed for model evaluation.
- TensorBoard is used for monitoring training progress.

### Elmo Embedding Model

- An Elmo embedding model is built using TensorFlow and Keras.
- The model architecture includes an ElmoEmbeddingLayer, a Dense layer, and a binary output layer with sigmoid activation.
- The model is trained with binary cross-entropy loss and Adam optimizer.

## Model Evaluation

- The trained models are evaluated on the test dataset.
- Metrics such as accuracy, precision, recall, and F1 score are computed for each model.
- Results are compared to assess the performance of each model.

## Requirements

- Python 3.5
- TensorFlow
- Keras
- nltk
- allennlp
- keras_metrics
