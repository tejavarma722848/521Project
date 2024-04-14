# Sentiment Analysis using Elmo Embeddings

## Introduction

This project demonstrates sentiment analysis using Elmo embeddings, a pre-trained deep learning model for extracting contextualized word representations. Elmo embeddings capture word meanings in the context of the entire sentence, allowing for more nuanced representations of text data.

## Prerequisites

Ensure you have the following dependencies installed:

- TensorFlow
- TensorFlow Hub
- Keras
- NLTK
- Allennlp
- WordCloud
- Matplotlib
- NumPy
- Pandas

## Dataset

The dataset used for sentiment analysis is located at `/Users/teja/Downloads/Twitter_Data.csv`. It contains text data and corresponding sentiment labels (0 for negative, 1 for positive). The dataset is preprocessed to balance the number of negative and positive samples.

## Preprocessing

- Converting Ratings to Sentiments: The ratings (0 and 1) in the Score column are mapped to sentiments (negative and positive), and a new Sentiment column is created.
- Text Preprocessing: Text data is preprocessed by converting characters to lowercase, removing HTML tags, punctuation, numbers, and stopwords. The words are tokenized, lemmatized, and vectorized using Elmo embeddings.

## Model Architecture

The sentiment analysis model architecture consists of:

- Elmo Embedding Layer: A custom Keras layer is defined for Elmo embeddings, allowing the model to use contextualized word representations.
- Fully Connected Layers: Dense layers with ReLU activation functions.
- Output Layer: A dense layer with a sigmoid activation function for binary classification.

The model is compiled with binary cross-entropy loss and the Adam optimizer.

## Training

The model is trained on the preprocessed dataset with a validation split of 20%, using 4 epochs and a batch size of 32.

## Evaluation

The model is evaluated on the test set using metrics such as accuracy, precision, recall, and F1 score.

## Results

The model achieves the following performance metrics on the test set:

- Accuracy: [Accuracy Score]
- Precision: [Precision Score]
- Recall: [Recall Score]
- F1 Score: [F1 Score]
