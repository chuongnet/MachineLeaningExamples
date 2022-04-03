import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random
import numpy as np
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
ignore_letters = ['!', '?', ',', '.']
model = load_model('model/chatbot_model.h5')


def clean_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens if w not in ignore_letters]
    return tokens


def bag_of_words(sentence, words, show_detail=True):
    tokens = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in tokens:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_detail:
                    print('found in bag: %s' % w)
    return np.array(bag)


def predict_class(sentence):
    p = bag_of_words(sentence, words, show_detail=False)
    res = model.predict(np.array([p]))[0]
    Error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > Error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    predicts = list()
    for r in results:
        predicts.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return predicts


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_intents = intents_json['intents']
    for i in list_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Creating tkinter GUI
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatBox.config(state=DISABLED)
# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set
# Create Button to send message

SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0, bg="#f9a602",
                    activebackground="#3c9d9b", fg='#000000', command=send)

# Create the box to enter message
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()

