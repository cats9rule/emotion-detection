import random
from keras.models import load_model
from keras.utils import plot_model

import preprocessing as pp
import train

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

def generateQuote(emotion: str):
    with open(f'quotes/{emotion}.txt', 'r', encoding='utf8') as file:
        quotes = file.readlines()
        index = random.randint(0, len(quotes)-1)
        return quotes[index].replace('\n', '')


def main():
    try:
        model = load_model('model\emotions.h5')
    except IOError:
        print("Model unavailable on disk. Making the model...")
        model = train.trainModel()
        plot_model(model=model, to_file='model\model.png', show_layer_names=False, show_shapes=True, show_layer_activations=True)
    
    while True:
        topredict = list()
        print("\n\n> Please enter a sentence: ")
        text = input()
        text = pp._normalizeText(text)
        topredict.append(text)
        topredict = pp.textToSequences(topredict)
        result = model.predict(topredict, 1, 0)
        emotion = ""
        max = 0
        for index, value in enumerate(result[0]):
            if value > max:
                max = value
                emotion = resolve_emotion(index)

        print("\n> Sentence: " + text + "\n  > Predicted: " + emotion)
        print(f'\n {generateQuote(emotion)}')



if __name__ == "__main__":
    main()