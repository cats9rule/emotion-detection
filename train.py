import preprocessing as pp

import pandas as pd
import io
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def trainModel():
    train = pd.read_csv('data/train.txt', sep=';', names=['text', 'sentiment'])
    test = pd.read_csv('data/test.txt', sep=';', names=['text', 'sentiment'])
    val = pd.read_csv('data/val.txt', sep=';', names=['text', 'sentiment'])
    _saveClassDistribution(train)

    xtrain, ytrain = pp.preprocess(train)
    xtest, ytest = pp.preprocess(test)
    xval, yval = pp.preprocess(val)

    # pp.makeTokenizer(xtrain)
    # xtrain = pp.textToSequences(xtrain)
    # xtest = pp.textToSequences(xtest)
    # xval = pp.textToSequences(xval)

    pp.make_word_embedding(xtrain)
    xtrain = pp.text_to_embedding(xtrain)
    xtest = pp.text_to_embedding(xtest)
    xval = pp.text_to_embedding(xval)

    model = _makeModel()
    hist = model.fit(xtrain, ytrain, epochs=15, validation_data=(xval, yval))
    model.save('model\emotions-wv.h5')
    print("Model saved on disk.")

    _testModel(model, xtest, ytest)
    _saveFigures(hist)
    plot_model(model=model, to_file='model\model-wv.png', show_layer_names=False, show_shapes=True, show_layer_activations=True)

    return model

def _makeModel():
    model = Sequential()
    #model.add(Embedding(input_dim=128, output_dim=64, input_length=80))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(200)))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(6, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.build((None, 80, 100))
    print(model.summary())
    return model

def _testModel(model, xtest, ytest):
    print("Testing the model...")
    print("The result obtained is:\n")
    test = pd.read_csv('data/test.txt', sep=';', names=['text', 'sentiment'])
    xtest, ytest = pp.preprocess(test)
    xtest = pp.text_to_embedding(xtest)
    model.evaluate(xtest, ytest)


    yprediction = model.predict(xtest)
    yprediction = np.argmax(yprediction, axis=1)
    ytest = np.argmax(ytest, axis=1)
    result = confusion_matrix(ytest, yprediction, normalize='pred')
    print(result)

    classes = ["joy", "anger", "love", "sadness", "fear", "surprise"]
    df_cfm = pd.DataFrame(result, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig("evaluation\cfm-wv.png")
    print("\nConfusion matrix saved on disk.\n")

    getClassificationReport(model)


def getClassificationReport(model):
    test = pd.read_csv('data/test.txt', sep=';', names=['text', 'sentiment'])
    xtest, ytest = pp.preprocess(test)
    xtest = pp.text_to_embedding(xtest)
    yprediction = model.predict(xtest)
    yprediction = np.argmax(yprediction, axis=1)
    ytest = np.argmax(ytest, axis=1)
    report = classification_report(ytest, yprediction)
    print(report)
    with io.open('evaluation/classificationreport-wv.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def _saveFigures(hist):
    plt.figure(figsize=(15, 10))
    plt.plot(hist.history['accuracy'], c='b', label='train')
    plt.plot(hist.history['val_accuracy'], c='r', label='validation')
    plt.title("Model Accuracy vs Epochs")
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    plt.legend(loc='lower right')
    plt.savefig('evaluation/accuracy-vw.jpg')

    plt.figure(figsize=(15, 10))
    plt.plot(hist.history['loss'], c='orange', label='train')
    plt.plot(hist.history['val_loss'], c='g', label='validation')
    plt.title("Model Loss vs Epochs")
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.legend(loc='upper right')
    plt.savefig('evaluation\loss-wv.jpg')

def _saveClassDistribution(dataset):
    plt.figure(figsize=(15, 10))
    pie = dataset.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')
    pie.figure.set_size_inches(15, 10)
    pie.figure.legend(loc='lower right')
    pie.figure.savefig("evaluation/classdistribution.jpg")
