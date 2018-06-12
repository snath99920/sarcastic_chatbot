# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:03:03 2018

@author: VISHAL-PC
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle

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
        
#ques_input = ques_input[0:50]        
#ans_input = ans_input[0:50]



t = Tokenizer(filters='')
t.fit_on_texts(ans_input)
encoded_docs = t.texts_to_sequences(ans_input)
#print(encoded_docs)
word_indexes = t.word_index
#print(t.word_index)

#Decreasing decoder vocabulary

total_vocab = 20000

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
decoder_input_data = np.zeros(
    (len(ques_input), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(ques_input), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


for i in range(0,len(encoded_docs)):
        for l in range(0,len(encoded_docs2[i])):
            encoder_input_data[i, l] = encoded_docs2[i][l]
        for l in range(0,len(encoded_docs[i])):
            decoder_input_data[i,l,encoded_docs[i][l]-1] = 1.
            if(l > 0):
                decoder_target_data[i,l-1,encoded_docs[i][l]-1] = 1.


pickleFile = open('pickledData', 'wb')

pickle.dump(encoded_docs, pickleFile)  
pickle.dump(word_indexes,pickleFile)  
pickle.dump(encoded_docs2,pickleFile)                
pickle.dump(word_indexes2,pickleFile)                
pickle.dump(embeddings_index,pickleFile)                
pickle.dump(max_encoder_seq_length,pickleFile)                
pickle.dump(max_decoder_seq_length,pickleFile)                
pickle.dump(num_encoder_tokens,pickleFile)                
pickle.dump(num_decoder_tokens,pickleFile)                
pickle.dump(embedding_matrix,pickleFile)                
pickle.dump(encoder_input_data,pickleFile)                
pickle.dump(decoder_input_data,pickleFile)                
pickle.dump(decoder_target_data,pickleFile) 
pickle.dump(ques_input,pickleFile)                
pickle.dump(ans_input,pickleFile)               

pickleFile.close()