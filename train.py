import preprocessing as pp

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt


def makeModel():
    train = pd.read_csv('data/train.txt', sep=';', names=['text', 'sentiment'])
    test = pd.read_csv('data/test.txt', sep=';', names=['text', 'sentiment'])
    val = pd.read_csv('data/val.txt', sep=';', names=['text', 'sentiment'])

    xtrain, ytrain = pp.preprocess(train)
    xtest, ytest = pp.preprocess(test)
    xval, yval = pp.preprocess(val)

    pp.makeTokenizer(xtrain)
    xtrain = pp.textToSequences(xtrain)
    xtest = pp.textToSequences(xtest)
    xval = pp.textToSequences(xval)

    model = Sequential()
    model.add(Embedding(15212, 64, input_length=80))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Dense(6, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model making done....")
    print(model.summary())

    print("Running the model....")
    hist = model.fit(xtrain, ytrain, epochs=15, validation_data=(xval, yval))

    print("Saving the model into the disk....")
    model.save('model\emotions.h5')
    print("Model saved into the disk....")

    print("Plotting the figures....")
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
    print("Figures saved in the disk....")

    # testing the model
    print("Testing the model....")
    print("The result obtained is...\n")
    model.evaluate(xtest, ytest)