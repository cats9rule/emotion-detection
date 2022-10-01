import preprocessing as pp

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
#from keras.metrics import Recall, Accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def trainModel():
    train = pd.read_csv('data/train.txt', sep=';', names=['text', 'sentiment'])
    test = pd.read_csv('data/test.txt', sep=';', names=['text', 'sentiment'])
    val = pd.read_csv('data/val.txt', sep=';', names=['text', 'sentiment'])

    plt.figure(figsize=(15, 10))
    pie = train.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')
    pie.figure.set_size_inches(15, 10)
    pie.figure.legend(loc='lower right')
    pie.figure.savefig("evaluation/classdistribution.jpg")

    xtrain, ytrain = pp.preprocess(train)
    xtest, ytest = pp.preprocess(test)
    xval, yval = pp.preprocess(val)

    pp.makeTokenizer(xtrain)
    xtrain = pp.textToSequences(xtrain)
    xtest = pp.textToSequences(xtest)
    xval = pp.textToSequences(xval)

    model = makeModel()

    print("Running the model...")
    hist = model.fit(xtrain, ytrain, epochs=15, validation_data=(xval, yval))

    print("Saving the model on disk....")
    model.save('model\emotions.h5')
    print("Model saved on disk.")

    testModel(model, xtest, ytest)
    saveFigures(hist)

    return model

def makeModel():
    model = Sequential()
    model.add(Embedding(15212, 64, input_length=80))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    #model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(6, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model making done.")
    print(model.summary())
    return model

def testModel(model, xtest, ytest):
    print("Testing the model....")
    print("The result obtained is:\n")
    model.evaluate(xtest, ytest)

    yprediction = model.predict(xtest)
    yprediction = np.argmax(yprediction, axis=1)
    ytest = np.argmax(ytest, axis=1)
    result = confusion_matrix(ytest, yprediction, normalize='pred')
    print(result)

    classes = ["joy", "sadness", "anger", "fear", "surprise", "love"]
    df_cfm = pd.DataFrame(result, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("evaluation\cfm.png")
    print("\nConfusion matrix saved on disk.\n")

def saveFigures(hist):
    plt.figure(figsize=(15, 10))
    plt.plot(hist.history['accuracy'], c='b', label='train')
    plt.plot(hist.history['val_accuracy'], c='r', label='validation')
    plt.title("Model Accuracy vs Epochs")
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    plt.legend(loc='lower right')
    plt.savefig('evaluation/accuracy.jpg')

    plt.figure(figsize=(15, 10))
    plt.plot(hist.history['loss'], c='orange', label='train')
    plt.plot(hist.history['val_loss'], c='g', label='validation')
    plt.title("Model Loss vs Epochs")
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.legend(loc='upper right')
    plt.savefig('evaluation\loss.jpg')