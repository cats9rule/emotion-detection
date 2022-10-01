import re
import io
import json
import string
import nltk
from nltk.tokenize import word_tokenize
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences


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
