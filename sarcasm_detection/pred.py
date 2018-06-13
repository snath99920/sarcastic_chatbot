#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:07:43 2018

@author: nikhil
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import pandas as pd


data = pd.read_csv('cleaned_data.csv', sep='\t',header=None)
X = data.iloc[:2000,1:2].values
Xdata = []

labels = data.iloc[:2000,0:1].values

for i in range(len(X)):
    Xdata.append(str(X[i]))
    
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(Xdata)

#--------------------------Prediction from model.json--------------------------

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


twt = ['I prefer the Anakin and Padme tiger scene in Ep 2']
#vectorizing the tweet by the pre-fitted tokenizer instance
#
## integer encode the documents
pred_docs = t.texts_to_sequences(twt)

#####change max length according to input
twt = pad_sequences(pred_docs, maxlen=112, padding='post')

output = loaded_model.predict_classes(twt)