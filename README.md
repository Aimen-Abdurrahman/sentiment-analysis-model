# sentiment-analysis-model

This repository contains a PyTorch-based sentiment analysis model that preprocesses textual data, trains a neural network, and predicts sentiment for a set of test sentences.

## Files Overview
data_preprocessing.py: Handles the encoding of sentences and prepares the dataset for training.
model_class.py: Defines the SentimentPredictor model architecture.
train_model.py: Loads data, preprocesses it, initializes the model, trains it, and tests it on sample sentences.

## Prerequisites
PyTorch
Pandas

## Dataset
The model uses the following dataset for training: (https://github.com/gptandchill/sentiment-analysis/blob/main/EcoPreprocessed.csv).

## Usage
Data Preprocessing: The data_preprocessing.py script is responsible for extracting vocabulary, encoding sentences, and preparing the dataset for training.

Model Definition: The model_class.py script contains the SentimentPredictor class that defines the architecture of the model, including an embedding layer and a linear layer.

Training the Model: The train_model.py script combines the data preprocessing, model initialization, training processes and evaluates the model's performance based on predefined test sentences.

## Sample Test Cases
The model evaluates on the following test sentences:

1. "awesome sale"
2. "bad movie"
3. "worst wifi ever"
4. "not good movie"
The output will display the model's predicted sentiment for each test case (1 is positive, -1 is negative).

## Training Details
Loss Function: Mean Squared Error Loss (nn.MSELoss)
Optimizer: Adam Optimizer
Number of Epochs: 1000
Batch Size: 64
Embedding Dimension: 256
