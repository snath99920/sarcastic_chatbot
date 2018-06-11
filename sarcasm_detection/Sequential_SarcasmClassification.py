#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:18:20 2018

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
from keras.callbacks import ModelCheckpoint

pickleFile = open('pickleData','rb')
padded_docs = pickle.load(pickleFile)
embedding_matrix = pickle.load(pickleFile)
labels = pickle.load(pickleFile)
max_length = pickle.load(pickleFile)
vocab_size = pickle.load(pickleFile)


# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# fit the model
model.fit(padded_docs, labels, validation_split=0.33, batch_size=32 ,epochs=10, callbacks=callbacks_list, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)



# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# =================Prediction ====================================
# 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
# 
# twt = ['im curious']
# #vectorizing the tweet by the pre-fitted tokenizer instance
# twt = t.texts_to_sequences(twt)
# #padding the tweet to have exactly the same shape as `embedding_2` input
# twt = pad_sequences(twt, maxlen=max_length, dtype='int32', value=0)
# 
# sentiment = loaded_model.predict(twt,batch_size=1,verbose = 2)[0]
# if(np.argmax(sentiment) == 0):
#     print("negative")
# elif (np.argmax(sentiment) == 1):
#     print("positive")
# =============================================================================
