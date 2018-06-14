# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:02:43 2018

@author: VISHAL-PC
"""

#import numpy as np
#from keras.models import model_from_json
from keras.models import Model
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Concatenate

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


'''
ques_input = []
ans_input = []

with open('encoder.txt') as f:
    lines = f.read().split('\n')

ques_lines = lines

with open('decoder.txt') as f:
    lines = f.read().split('\n')

ans_lines = lines

for i in range(len(ans_lines)):
    sen1 = ques_lines[i].split()
    sen2 = ans_lines[i].split()
    
    if(len(sen1)<= 20):
        
        if(len(sen2)<= 20):
            
            ques_input.append(ques_lines[i])
            ans_input.append('<sos> '+ans_lines[i]+' <eos>')
        
ques_input = ques_input[0:50]        
ans_input = ans_input[0:50]



t = Tokenizer(filters='')
t.fit_on_texts(ans_input)
encoded_docs = t.texts_to_sequences(ans_input)
#print(encoded_docs)
word_indexes = t.word_index
#print(t.word_index)

#Decreasing decoder vocabulary

total_vocab = 20

reverse_word_index = dict(
    (i, word) for word, i in word_indexes.items())
    
word_count = t.word_counts
sorted_d = sorted(word_count.items(), key=lambda x: x[1])
vocab_dict = {}
j=1
for i in range(len(sorted_d)-total_vocab,len(sorted_d)):
    #print(i)
    vocab_dict[sorted_d[i][0]]=j
    j = j+1
en_docs = []
for i in range(len(encoded_docs)):
    sent = []
    for j in range(len(encoded_docs[i])):
        if(vocab_dict.get(reverse_word_index[encoded_docs[i][j]])):
            sent.append(vocab_dict[reverse_word_index[encoded_docs[i][j]]])
    en_docs.append(sent)
encoded_docs = en_docs
word_indexes = vocab_dict  



t2 = Tokenizer(filters='')
t2.fit_on_texts(ques_input)
encoded_docs2 = t2.texts_to_sequences(ques_input)
#print(encoded_docs2)
word_indexes2 = t2.word_index
#print(t2.word_index)


embeddings_index = dict()
#reverse_embeddings_index = dict()
f = open('glove.6B.200d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


max_encoder_seq_length = max([len(encoded_docs2[i]) for i in range(0,len(encoded_docs2))])
max_decoder_seq_length = max([len(encoded_docs[i]) for i in range(0,len(encoded_docs))])
num_encoder_tokens = len(word_indexes2)
num_decoder_tokens = len(word_indexes)



print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)



embedding_matrix = np.zeros((num_encoder_tokens+1, 200))
for word, i in word_indexes2.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
encoder_input_data = np.zeros(
    (len(ques_input), max_encoder_seq_length),
    dtype='float32')        


for i in range(0,len(encoded_docs)):
        for l in range(0,len(encoded_docs2[i])):
            encoder_input_data[i, l] = encoded_docs2[i][l]

'''
latent_dim = 256
embedding_layer = Embedding(num_encoder_tokens+1,200,weights=[embedding_matrix],
                            input_length=max_encoder_seq_length,
                            trainable=False)

encoder_inputs = Input(shape=(None, ))
encoder_embedding = embedding_layer(encoder_inputs)
'''encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedding)'''
encoder = Bidirectional(LSTM(latent_dim, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embedding)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#model.load_weights("weights-improvement-07-0.03.hdf5")
model.load_weights("weights.best.hdf5")
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])    
    
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim*2,))
decoder_state_input_c = Input(shape=(latent_dim*2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

encoder_model.compile(optimizer='Adam', loss='categorical_crossentropy')
decoder_model.compile(optimizer='Adam', loss='categorical_crossentropy')
    
encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')