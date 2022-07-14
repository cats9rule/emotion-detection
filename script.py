import numpy as np
import pandas as pd
import string
import re
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

print("All the modules imported....")

train = pd.read_csv('train.txt', sep=';', names=['text', 'sentiment'])
test = pd.read_csv('test.txt', sep=';', names=['text', 'sentiment'])
val = pd.read_csv('val.txt', sep=';', names=['text', 'sentiment'])

print('Dataset loaded....')


def cleaning(text):
    text = text.lower()
    pattern = re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text


def resolve_emotion(index):
    if index == 0:
        return "joy"
    if index == 1:
        return "anger"
    if index == 2:
        return "love"
    if index == 3:
        return "sadness"
    if index == 4:
        return "fear"
    if index == 5:
        return "surprise"
    return ""


print("Cleaning is taking place....")
train['text'] = train['text'].map(cleaning)
print("Training done....")
test['text'] = test['text'].map(cleaning)
print("Testing done...")
val['text'] = val['text'].map(cleaning)
print("Val done....")
print("Cleaning is done...")

print("Data preprocessing is started....")
xtrain = train['text'].values
xtest = test['text'].values
xval = val['text'].values

train['sentiment'] = train['sentiment'].replace(
    {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5})
test['sentiment'] = test['sentiment'].replace(
    {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5})
val['sentiment'] = val['sentiment'].replace(
    {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5})

ytrain, ytest, yval = train['sentiment'].values, test['sentiment'].values, val['sentiment'].values

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
yval = to_categorical(yval)

# converting to text to sequences
tokenizer = Tokenizer(15212, lower=True, oov_token='UNK')
tokenizer.fit_on_texts(xtrain)
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)
xval = tokenizer.texts_to_sequences(xval)

# xtrain = pad_sequences(xtrain, maxlen=80, padding='post')
# xtest = pad_sequences(xtest, maxlen=80, padding='post')
# xval = pad_sequences(xval, maxlen=80, padding='post')
# print("Data preprocessing is over....")


# print("Making the model....")
# model = Sequential()
# model.add(Embedding(15212, 64, input_length=80))
# model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(64, return_sequences=True)))
# model.add(Bidirectional(LSTM(128)))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Dense(6, activation="softmax"))
# model.compile(optimizer='adam', loss='categorical_crossentropy',
#               metrics=['accuracy'])
# print("Model making done....")
# print(model.summary())

# print("Running the model....")
# hist = model.fit(xtrain, ytrain, epochs=15, validation_data=(xval, yval))

# print("Saving the model into the disk....")
# model.save('emotions.h5')
# print("Model saved into the disk....")

model = load_model('emotions.h5')

# print("Plotting the figures....")
# plt.figure(figsize=(15, 10))
# plt.plot(hist.history['accuracy'], c='b', label='train')
# plt.plot(hist.history['val_accuracy'], c='r', label='validation')
# plt.title("Model Accuracy vs Epochs")
# plt.xlabel("EPOCHS")
# plt.ylabel("ACCURACY")
# plt.legend(loc='lower right')
# plt.savefig('accuracy.jpg')


# plt.figure(figsize=(15, 10))
# plt.plot(hist.history['loss'], c='orange', label='train')
# plt.plot(hist.history['val_loss'], c='g', label='validation')
# plt.title("Model Loss vs Epochs")
# plt.xlabel("EPOCHS")
# plt.ylabel("LOSS")
# plt.legend(loc='upper right')
# plt.savefig('loss.jpg')
# print("Figures saved in the disk....")

# # testing the model
# print("Testing the model....")
# print("The result obtained is...\n")
# model.evaluate(xtest, ytest)

while True:
    topredict = list()
    print("\n\n  > Please enter a sentence: ")
    text = input()
    text = cleaning(text)
    topredict.append(text)
    runonme = tokenizer.texts_to_sequences(topredict)
    runonme = pad_sequences(runonme, maxlen=80, padding='post')

    result = model.predict(runonme, 1, 0)

    emotion = ""
    max = 0

    for index, value in enumerate(result[0]):
        if value > max:
            max = value
            emotion = resolve_emotion(index)

    print("\n  > Sentence: " + text + "\n  > Predicted: " + emotion)
