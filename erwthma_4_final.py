#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from elasticsearch import helpers, Elasticsearch

from sklearn.cluster import KMeans

from itertools import product

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

es = Elasticsearch()


# In[ ]:


df_movies = pd.read_csv(r".\data\movies.csv",engine="python")
df_ratings = pd.read_csv(r".\data\ratings.csv",engine="python").drop("timestamp", axis = 1)

#load the ratings
df = pd.read_csv(r".\data\ratings.csv",engine="python")
df2 = df.groupby('movieId')

#hold the mean rating for each movie
df3 = df2['rating'].mean()
df3 = pd.Series.to_frame(df3)


# In[ ]:


temp = 1
while (temp != 0):
    
    #Asking from the user to insert the movie
    movie = input("Insert the movie you want\n")
    
    #Asking for user ID and checking its validity
    user = input("Who is watching?\n")
    while(not user.isdigit()):user=input("Please give a valid user ID\n")
        
    user = int(user)
    
    #Clearing result dataframe after every search
    rows = []
    res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(movie)}}}, size = 1000)
    
    #to get only the value of total hits
    hits = res['hits']['total']['value']
    print("Total number of hits",hits)
    i = 0
    if hits != 0:
        for hit in res['hits']['hits']:
            i = i+1
            
            #Extracting data for each hit
            title = hit['_source'].get('title')
            movie = hit['_source'].get('movieId')
            BM25_score = hit['_score']
            movie = int(movie)
            
            #Finding average rating for all the movies while checking if the movie has no ratings
            try:
                average_rating = df3.loc[movie]
                average_rating = pd.Series.to_frame(average_rating)
                avrrat = average_rating.values
            except KeyError:
                avrrat = None
                
            #Finding user's rating for the movies if they exist
            user_rating = df["rating"][(df["movieId"] == movie) & (df["userId"] == user)]
            user_rating = pd.Series.to_frame(user_rating)
            usrrat = user_rating.values
            #If they do not, we replace them with the ones that were predicted
            if (usrrat.size == 0) :
                with open(r".\data\pickles\user_{}_predicted_data.p".format(user), "rb") as f:
                    user_predicted_data = pickle.load(f)
                usrrat = float(user_predicted_data.loc[user_predicted_data['movieId'] == movie, 'rating'].values)
            
            #Extracting the final rating score while checking if average ratings exist
            if (avrrat == None) :
                final_rating= BM25_score + float(usrrat)
            else : final_rating = BM25_score + float(avrrat) + float(usrrat)
                
            rows.append([title, movie, final_rating])
            
        #Constructing the Dataframe which shows the results
        df_new = pd.DataFrame(rows, columns = ["title  year ", "movieId", "final rating"])
        df_new = df_new.sort_values('final rating', ascending = False).reset_index()
        df_new = df_new.drop(["index"], axis=1)
        print(df_new)
    else:
        print("No such result\n")
        
    #Asking the user for a new search, thus continuing the loop
    temp = input("If you want to exit press 0\n")
    if (temp.isdigit()) : 
        temp = int(temp)
        if (int(temp) == 0) : print("Thank you! See you next time!")

