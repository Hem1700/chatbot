import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizer import SGD
import random
 

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Iterating through the patterns and tokenize the sentence

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each owrd
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # add documents in the corpus
        documents.append((w,intent['tag']))

        # add to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])



# lemmatizing each word , lower each word and remove duplicted

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sorting classes
classes = sorted(list(set(classes)))

# documents is combination between patterns and intents
print(len(documents), "documents")

# classes is equal to intents
print(len(classes),"classes", classes)

#words = all words, vocublary
print(len(words), "unique lemmatized words", words)


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl','wb'))





