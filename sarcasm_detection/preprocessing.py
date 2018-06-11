#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:26:59 2018

@author: nikhil
"""

import pickle
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import pandas as pd
import numpy as np
from keras.layers import LSTM
from keras.models import model_from_json


data = pd.read_csv('cleaned_data.csv', sep='\t',header=None)
X = data.iloc[:80000,1:2].values
Xdata = []

labels = data.iloc[:80000,0:1].values

for i in range(len(X)):
    Xdata.append(str(X[i]))
    
    
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(Xdata)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(Xdata)
print(encoded_docs)
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
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


pickleFile = open('pickleData','wb')

pickle.dump(padded_docs,pickleFile)
pickle.dump(embedding_matrix, pickleFile)
pickle.dump(labels,pickleFile)
pickle.dump(max_length,pickleFile)
pickle.dump(vocab_size,pickleFile)

pickleFile.close()






