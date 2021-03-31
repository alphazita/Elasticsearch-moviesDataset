#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from elasticsearch import helpers, Elasticsearch
from sklearn.cluster import KMeans
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer

import csv
import pandas as pd
import numpy as np
import math
import re
import pickle


# In[ ]:


es = Elasticsearch()

df_movies = pd.read_csv(r".\data\movies.csv",engine="python", encoding="utf-8")
df_ratings = pd.read_csv(r".\data\ratings.csv",engine="python")


# In[ ]:


df_movies['title'] = df_movies['title'].map(lambda x: re.sub(r'([^\s\w]|_)+', '', x))
df_movies['title'] = df_movies['title'].str.lower()
vectorizer = CountVectorizer()
corpus = list(df_movies['title'].values)

wordvec = vectorizer.fit_transform(corpus)

wordvec = wordvec.toarray()


words = vectorizer.get_feature_names()
print("number of words in movie names: ", len(words))


# In[ ]:


corpus = list(df_movies['title'].values)
embeddings_index = {}
f = open(r'.\data\glove.6B\glove.6B.100d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors in GloVe .txt file.' % len(embeddings_index))


# In[ ]:


templist = []
for i in range (100):
    templist.append(str(i))
    
templist2 =[]
for i in range (9124):
    templist2.append(i)

df_v= pd.DataFrame(index = templist2, columns = templist)

print ("Words from movie titles not found in GloVe .txt file:")

for movieindex in range (9124):
    temp = corpus[movieindex].split()
    sumvec = 0
    for i in temp:
        try:
            sumvec += embeddings_index[i]
        except KeyError:
            print (i)
            pass
    sumvec = sumvec.reshape(1,100)
    
    for i in range (100):
        df_v.iloc[movieindex, i] = sumvec[0][i]
    
print("Titles in vectors Dataframe is ready!")


# In[ ]:


#one hot encoding movie genres
temp2 = df_movies['genres'].str.get_dummies()
temp2.columns = ['Genre_' + str(col) for col in temp2.columns ]

genre_of_movie = df_v.merge(temp2, left_index=True, right_index=True)
genre_of_movie.insert(120, "movieId", df_movies['movieId'])
df_ohe = genre_of_movie
print (df_ohe)


# In[ ]:


#list of ratings in order to one hot encode them in Y_train
ratings_lst = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# map each rating to an integer
mapping = {}
for x in range(len(ratings_lst)):
    mapping[ratings_lst[x]] = x


# In[ ]:


#here we create the data and 'save' them on pickles
print ("This may take a while...")
for userid in df_ratings['userId'].unique():
    print("user-----------------------------------------------------------------------------",userid,"\n")
    user_data = df_ohe.merge(df_ratings[df_ratings['userId'] == userid], on='movieId', how = 'outer')
    
    #create prediction data
    user_prediction_data = user_data[user_data['rating'].isnull()]
    user_prediction_data.drop('userId',axis=1,inplace=True)
    user_prediction_data.drop('timestamp',axis=1,inplace=True)
    
    #create training data
    user_training_data = user_data[~user_data['rating'].isnull()]
    user_training_data.drop('userId',axis=1,inplace=True)
    user_training_data.drop('timestamp',axis=1,inplace=True)
    
    #set the index
    user_training_data.set_index('movieId', inplace=True)
    user_prediction_data.set_index('movieId', inplace=True)
    
    #dump the data on pickles
    with open(r".\data\pickles\user_{}_training_data.p".format(userid), "wb") as f:
        pickle.dump(user_training_data, f)
    
    
    with open(r".\data\pickles\user_{}_prediction_data.p".format(userid), "wb") as f:
        pickle.dump(user_prediction_data, f)
        
print("Pickles are ready!")


# In[ ]:


for user in df_ratings['userId'].unique():
    print("user --------------------> ", user)

    with open(r".\data\pickles\user_{}_training_data.p".format(user), "rb") as f:
        user_training_data = pickle.load(f)

    with open(r".\data\pickles\user_{}_prediction_data.p".format(user), "rb") as g:
        user_prediction_data = pickle.load(g)

    X_train = user_training_data.loc[:, user_training_data.columns != 'rating']
    Y_train = user_training_data['rating']

    X_test = user_prediction_data.loc[:, user_prediction_data.columns != 'rating']
    Y_test = user_prediction_data['rating']

    one_hot_encode = []

    for r in Y_train:
        arr = list(np.zeros(len(ratings_lst), dtype = int))
        arr[mapping[r]] = 1
        one_hot_encode.append(arr)

    #print(type(one_hot_encode))

    Y_train = np.array(one_hot_encode)


    # define the keras model
    model = Sequential()
    model.add(Dense(500, activation='relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    # compile the keras model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

    # fit the keras model on the dataset
    model.fit(X_train, Y_train, epochs=30, batch_size=5, verbose=0)

    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, Y_train)
    print('Accuracy: %.2f' % (accuracy*100))

    Y_pred = model.predict(X_test)
    rat_trans_lst = []
    for i in range(Y_pred.shape[0]):
        tmp = ((np.argmax(Y_pred[i]))+1)/2
        rat_trans_lst.append(tmp)
    user_prediction_data['rating'] = rat_trans_lst
    user_predicted_data = user_prediction_data.append(user_training_data).reset_index()
    user_predicted_data = user_predicted_data[['movieId', 'rating']]
    
    with open(r".\data\pickles\user_{}_predicted_data.p".format(user), "wb") as k:
        pickle.dump(user_predicted_data, k)

