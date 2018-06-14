#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 13:18:20 2018

@author: nikhil
"""
import pickle
from keras.layers import Embedding, Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

pickleFile = open('pickleDetect','rb')
t = pickle.load(pickleFile)
padded_docs = pickle.load(pickleFile)
embedding_matrix = pickle.load(pickleFile)
labels = pickle.load(pickleFile)
max_length = pickle.load(pickleFile)
vocab_size = pickle.load(pickleFile)

print(max_length)

## create model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit the model
model.fit(padded_docs, labels, validation_split=0.33, batch_size=32 ,epochs=20, callbacks=callbacks_list, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")