#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from elasticsearch import helpers, Elasticsearch
import csv
es = Elasticsearch()


# In[ ]:


import pandas as pd
import numpy as np

#load the ratings

df = pd.read_csv(r".\data\ratings.csv",engine="python")
df2 = df.groupby('movieId')

#hold the mean rating for each movie

df3 = df2['rating'].mean()
df3 = pd.Series.to_frame(df3)


# In[ ]:


#2nd

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
            if (usrrat.size == 0) : usrrat = None
            
            #Extracting the final rating score while checking if user and average ratings exist
            if (usrrat == None) :
                if (avrrat == None) : final_rating = BM25_score
                else : final_rating = BM25_score + float(avrrat)
            else : final_rating = BM25_score + float(avrrat) + float(usrrat)
                
            rows.append([title, movie, final_rating])
    else:
        print("No such result\n")
        
    #Constructing the Dataframe which shows the results
    df_new = pd.DataFrame(rows, columns = ["title  year ", "movieId", "final rating"])
    df_new = df_new.sort_values('final rating', ascending = False).reset_index()
    df_new = df_new.drop(["index"], axis=1)
    print(df_new)
    
    #Asking the user for a new search, thus continuing the loop
    temp = input("Do you want to search for another movie? (0 = No)\n")
    if (temp.isdigit()) : 
        temp = int(temp)
        if (int(temp) == 0) : print("Thank you! See you next time!")

