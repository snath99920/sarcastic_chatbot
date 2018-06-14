#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:16:58 2018

@author: nikhil
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import asarray
from numpy import zeros
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import model_from_json

data = pd.read_csv('cleaned_data.csv', sep='\t',header=None)
X = data.iloc[:50,1:2].values
Xdata = []

labels = data.iloc[:50,0:1].values

for i in range(len(X)):
    Xdata.append(str(X[i]))
    
t = Tokenizer()
t.fit_on_texts(Xdata)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(Xdata)

max_length = max([len(encoded_docs[i]) for i in range (0,len(encoded_docs))])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# load the whole embedding into memory
embeddings_index = dict()
f = open('/media/nikhil/5204303204301B83/cs/IIIT/classification/glove/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)

model.add(LSTM(50, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(1,activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=10, batch_size=32, verbose=2)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


twt = ['you can?']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = t.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=max_length, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")