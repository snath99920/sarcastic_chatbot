#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:07:43 2018

@author: nikhil
"""

from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import pickle

pickleFile = open('pickleData','rb')
t = pickle.load(pickleFile)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


twt = ['I prefer the Anakin and Padme tiger scene in Ep 2']
pred_docs = t.texts_to_sequences(twt)
twt = pad_sequences(pred_docs, maxlen=112, padding='post')
output = loaded_model.predict_classes(twt)