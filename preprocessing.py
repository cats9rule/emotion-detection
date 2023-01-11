import re
import io
import json
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import gensim.downloader
from gensim.models import Word2Vec, KeyedVectors
from os import path

model = None

def _normalizeText(text: str) -> str:
    text = text.lower()
    text = _cleanAbbreviations(_cleanLinks(text))
    text = _removeSpecialChars(text)
    text = _removePunctuation(text)
    return text

def _cleanLinks(text):
    pattern = re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = pattern.sub('', text)
    return text

def _cleanAbbreviations(text):
    text = re.sub(r"i['\s]?m\s", "i am ", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"\su['\s]r\s", " you are ", text)
    text = re.sub(r"he['\s]?s\s", "he is ", text)
    text = re.sub(r"she['\s]?s", "she is", text)
    text = re.sub(r"that['\s]?s", "that is", text)
    text = re.sub(r"what['\s]?s", "what is", text)
    text = re.sub(r"where['\s]?s", "where is", text)
    text = re.sub(r"[\'\s]ll\s", " will ", text)
    text = re.sub(r"[\'\s]ll\s", " will ", text)
    text = re.sub(r"[\'\s]ve\s", " have ", text)
    text = re.sub(r"[\'\s]re\s", " are ", text)
    text = re.sub(r"[\'\s]d\s", " would ", text)
    text = re.sub(r"won['\s]?t", "will not", text)
    text = re.sub(r"don['\s]?t", "do not", text)
    text = re.sub(r"didn['\s]?t", "did not", text)
    text = re.sub(r"can['\s]?t", "can not", text)
    text = re.sub(r"couldn['\s]?t", "could not", text)
    text = re.sub(r"haven['\s]?t", "have not", text)
    text = re.sub(r"\sw(?:\s|$)", " with ", text)
    text = re.sub(r"\stbh\s", " to be honest ", text)
    return text

def _removeSpecialChars(text):
    text = re.sub(r"[@#$%^&*(){}/;`~<>+=-]", "", text)
    return text

def _removePunctuation(text):
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text

def preprocess(inputData: dict) -> tuple:
    inputData['sentiment'] = inputData['sentiment'].replace(
    {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5})
    yinput = to_categorical(inputData['sentiment'].values)
    inputData['text'] = inputData['text'].map(_normalizeText)
    xinput = inputData['text'].values
    return (xinput, yinput)

def makeTokenizer(xtrain: list):
    tokenizer = Tokenizer(15212, oov_token='UNK')
    tokenizer.fit_on_texts(xtrain)
    tokenizer_json = tokenizer.to_json()
    with io.open('preprocessing/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    return
    
def textToSequences(text: list):
    tokenizer = None
    with open('preprocessing/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        text = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=80, padding='post')
        return text

########### WORD2VEC ############

def make_word_embedding(xtrain: list):
    if not path.exists('preprocessing/modelpath.txt'):
        model_path = gensim.downloader.load('glove-twitter-100', return_path=True) #Word2Vec(sentences=xtrain, vector_size=128, window=5, min_count=1, workers=10)
        print(model_path)
        with io.open('preprocessing/modelpath.txt', 'w', encoding='utf-8') as f:
            f.write(model_path)
    return

def text_to_embedding(text: list):
    global model
    if model is None:
        model_path = ''
        with open('preprocessing/modelpath.txt') as f:
            model_path = f.read()
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
    
    words = set(model.index_to_key)
    #print("\n\nVector: " + str(model['angry']) + "\n\n\n")

    tokenized_text = []

    for sentence in text:
        tokenized_sentence = [model[i] for i in sentence if i in words]
        word_count = len(tokenized_sentence)
        if word_count > 80:
            tokenized_sentence = tokenized_sentence[:80]
        elif word_count < 80:
            tokenized_sentence.extend([[0.0] * 100] * (80 - word_count))
        tokenized_sentence = np.array(tokenized_sentence)
        #print(tokenized_sentence.shape)
        tokenized_text.append(tokenized_sentence)
    tokenized_text = np.array(tokenized_text)
    
    # tokenized_text = np.array([np.array(val) for val in tokenized_text])
    print("\n\n" + str(tokenized_text.shape))
    return tokenized_text


    # embed_list = list()
    # for sentence in text:
    #     embed_sentence = list()
    #     for word in sentence:
    #         embed_sentence.append(model.wv[word])
    #     embed_list.append(pad_sequences(embed_sentence, maxlen=80, padding='post'))
    
    # print("\n\n" + str(len(embed_sentence)) + "\n\n")
    # return embed_list
    #return pad_sequences(embed_list, maxlen=80, padding='post')