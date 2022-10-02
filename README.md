# Motivational Quote Generator Using Emotion Detection

This repository contains a motivational quote generator. 
The program uses a DNN model to predict the emotion of an input sentence. 
Then, based on the classification results, the program chooses a motivational quote and displays it in the console.

The model has 1 input, 1 output and 5 hidden layers (2 Bi-LSTM, 1 Dense, 2 Dropout). 

Text preprocessing uses regular expressions for text cleanup and Keras Tokenizer for word representation.

The model uses an Embedding input layer for creating dense vectors out of input data.

---

### Specifications 
Language: Python

Dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

The DNN model is based on an existing [project](https://github.com/Ankit152/Emotions-Detection-NLP).

