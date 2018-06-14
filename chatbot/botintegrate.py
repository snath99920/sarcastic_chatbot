# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:43:52 2018

@author: VISHAL-PC
"""
import numpy as np
from keras.models import model_from_json
#from keras.models import Model
from keras.models import load_model
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
#from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate
from keras.preprocessing.sequence import pad_sequences

import pickle

pickleFile = open('pickledData', 'rb') 
    
#encoded_docs = pickle.load(pickleFile)  
word_indexes = pickle.load(pickleFile) 
t2 = pickle.load(pickleFile) 
#encoded_docs2 = pickle.load(pickleFile)                
#word_indexes2 = pickle.load(pickleFile)                
#embeddings_index = pickle.load(pickleFile)                
max_encoder_seq_length = pickle.load(pickleFile)                
max_decoder_seq_length = pickle.load(pickleFile)                
num_encoder_tokens = pickle.load(pickleFile)                
num_decoder_tokens = pickle.load(pickleFile)                
embedding_matrix = pickle.load(pickleFile)                
#encoder_input_data = pickle.load(pickleFile)                
#decoder_input_data = pickle.load(pickleFile)                
#decoder_target_data = pickle.load(pickleFile)
#ques_input = pickle.load(pickleFile)                
#ans_input = pickle.load(pickleFile)

pickleFile.close()



encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')    
reverse_word_index = dict(
    (i, word) for word, i in word_indexes.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word_indexes['<sos>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    decoded_answer = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        #print(output_tokens[0, -1, :])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #print(sampled_token_index)
        sampled_word = reverse_word_index[sampled_token_index]
        sampled_word = sampled_word + " "
        decoded_sentence += sampled_word
        if(sampled_word == '<sos> '):
            decoded_answer = decoded_answer
        else:
            decoded_answer += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '<eos> ' or
           len(text_to_word_sequence(decoded_sentence)) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_answer



def predict(str):
    list = []
    list.append(str)
    str_docs=t2.texts_to_sequences(list)
    encoder_str_data = np.zeros((1,max_encoder_seq_length),dtype='float32')
    for l in range(0,len(str_docs[0])):
            encoder_str_data[0][l] = str_docs[0][l]
    decoded_sentence = decode_sequence(encoder_str_data)
    print('Predict Function:')
    print('Input sentence:', str)
    print('Decoded sentence:', decoded_sentence)


pickleDet = open('pickleDetect','rb')
t = pickle.load(pickleDet)
max_length = pickle.load(pickleDet)
pickleDet.close()
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def detect(str):
    twt=[str]
    #twt = ['I prefer the Anakin and Padme tiger scene in Ep 2']
    pred_docs = t.texts_to_sequences(twt)
    twt = pad_sequences(pred_docs, maxlen=max_length, padding='post')
    output = loaded_model.predict_classes(twt)
    print('\nDetectFunction:')
    if(output[0]==1):
        print('Sarcastic')
    else:
        print('Not Sarcastic')
  
predict('How is our little Find the Wench A Date plan progressing?')
detect('How is our little Find the Wench A Date plan progressing?')