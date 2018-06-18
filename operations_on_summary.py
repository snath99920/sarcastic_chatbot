
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lines = [line.rstrip() for line in open(r'C:/Users/Aravind/Downloads/summary.txt')]

titles = [line.rstrip() for line in open(r'C:\Users\Aravind\Downloads\Udemy,NLP\machine_learning_examples-master\machine_learning_examples-master\nlp_class\all_book_titles.txt')]
lines


# In[2]:


stopwords = set(w.rstrip() for w in open(r'C:\Users\Aravind\Downloads\Udemy,NLP\machine_learning_examples-master\machine_learning_examples-master\nlp_class\stopwords.txt'))


# In[3]:


def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


# In[9]:


word_index_map = {}   #tokenizes data
currentindex = 0
all_tokens =[]
all_lines = []
index_word_map = []
for line in lines:
    all_lines.append(line)
    tokens = my_tokenizer(line)
    all_tokens.append(tokens)
    
    
    
    for token in tokens:              #makes custom wordtoindex mapping
        if token not in word_index_map:
            word_index_map[token] = currentindex
            currentindex = currentindex+1
            index_word_map.append(token)
nltk.download('averaged_perceptron_tagger')  


   


# In[28]:


nltk.download('maxent_ne_chunker')


# In[30]:


nltk.download('words')


# In[50]:


postags = [[]]
singularnouns = [[]]
netemp = [[]]
i = 0
for tokenss in all_tokens:
     tokenstemp = nltk.pos_tag(tokenss)
     postags[i] = tokenstemp                    #pos taging
     chunk = nltk.ne_chunk(tokenss)
     netemp[i] = chunk                          #NE Recognition
     words = nltk.FreqDist(tokenss)             #Frequency Distribution of words to find most common words
        
        
     singularnouns[i] = [word for word,pos in postags[i] if pos == 'NN']
     i = i+1
        
        

words.most_common()


# In[51]:


print(netemp)            # weird results,need to clarify.Not getting proper NE tags
for chunkk in netemp:
    if hasattr (chunkk,'label'):
        print (chunkk.label(), " ".join(c[0] for c in chunkk.leaves()))


# In[52]:


def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] =  1
    
    return x


# In[53]:


N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D,N))
X.shape


# In[106]:


i = 0            #code for LSA .Again,a minor error possibly with data shape.Need to clarify

for token in all_tokens:
    print(token)
    X[:,i] = tokens_to_vector(token)
    i = i+1
svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:,0], Z[:,1])           #Plots everything
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
plt.show()   


# In[ ]:





# In[ ]:




