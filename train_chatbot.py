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


# create our training data
training = []

# create an empty array for output
output_empty = [0] * len(classes)

#training set, bag of words for each sentence
for doc in documents:
    #initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]

    #lemmatizinig each word - create base word , in attempt to represnt related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # creating baf og words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag(for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


