#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from elasticsearch import helpers, Elasticsearch
from sklearn.cluster import KMeans
from itertools import product
import csv
import pandas as pd
import numpy as np
import math
es = Elasticsearch()


# In[ ]:


df_movies = pd.read_csv(r".\data\movies.csv",engine="python")
df_ratings = pd.read_csv(r".\data\ratings.csv",engine="python").drop("timestamp", axis = 1)
all_genres = []

for i in range (len(df_movies)):
    temp = df_movies.loc[i, "genres"].split("|")
    for j in range (len(temp)) :
        if (temp[j] not in all_genres) :
            all_genres.append(temp[j])
#print (all_genres)


# In[ ]:


results = helpers.scan(es, index = 'ratings', query = {"query": {"match_all": {}}})
res_set = set()
for item in results:
    popo = item['_source']['userId']
    res_set.add(int(popo))    
#print(res_set)


# In[ ]:


df3 = df_ratings.drop("movieId", axis = 1)
df3.groupby('userId').first()
df3.drop("rating", axis = 1)

lstset = list(res_set)

data = {'userId': lstset}
user_df = pd.DataFrame.from_dict(data)
user_df = user_df.sort_values(by=['userId'])

#filling the dataframe with zeros
for i in range (len(all_genres)) :
    user_df[all_genres[i]] = pd.Series(0.0, index = user_df.index, dtype = 'float32')
#print(user_df)


# In[ ]:


#constructing the dataframe takes some time
for user in res_set:
    print("user(for1)-----------------------------------------------------------------------------",user,"\n")
    query_body_user = {"query": {"match": {"userId": user}}}
    res3 = es.search(index = "ratings", body = query_body_user, size = 1000)

    for hit in res3['hits']['hits']:
        movie = hit['_source'].get('movieId')
        
        query_body_genres = {"query": {"match": {"movieId": movie}}}
        res_gen = es.search(index = "movies", body = query_body_genres, size = 1000)
        genres = res_gen['hits']['hits'][0]['_source']['genres'].split('|')        
        
        query_body_rating = {"query": {"bool": {"should": [{"match" : { "userId": user}}, {"match" : {"movieId": movie}}]}}}
        res2 = es.search(index = "ratings", body = query_body_rating, size = 1000)
        rating = res2['hits']['hits'][0]['_source']['rating']
        rating = float(rating)        
        
        for genre in genres:
            if (user_df[genre][user-1] == 0.0):
                user_df[genre][user-1] = rating
            else :
                user_df[genre][user-1] = (user_df[genre][user-1] + rating) / 2.0
#print(user_df)


# In[ ]:


#we export the dataframe containing the average rating for each genre for all the users separately to a .csv file
print("Exporting user_df dataframe to csv ...")
user_df.to_csv (r'.\data\user_average_genre_ratings(no indices).csv', index = False, header=True)
print("Done!")


# In[ ]:


df_usavrt = pd.read_csv(r".\data\user_average_genre_ratings(no indices).csv",engine="python")
df_usavrt = df_usavrt.set_index('userId')
kmeans_1 = KMeans(n_clusters = 10, random_state = 21)
X = df_usavrt
predictions = kmeans_1.fit_predict(X)
centroids = kmeans_1.cluster_centers_
labels = kmeans_1.labels_
X['cluster'] = labels
df_clusters = X[['cluster']]

df_clusters = df_clusters.sort_values('cluster')
#df_clusters
print("Extracting df_clusters dataframe to csv ...")
df_clusters.to_csv (r'.\data\df_clusters.csv', index = False, header=True)
print("Done!")


# In[ ]:


l1 = list(df_ratings['userId'].unique())
l2 = list(df_movies['movieId'].unique())
usr_mov_clst = pd.DataFrame(list(product(l1, l2)), columns=['userId', 'movieId'])
usr_mov_clst.sort_values(by=['userId','movieId']).reset_index(inplace=True, drop=True)
usr_mov_clst = usr_mov_clst.merge(df_ratings, on = ['userId','movieId'], how='left')

usr_mov_clst = usr_mov_clst.merge(df_clusters, on='userId', how='left')

#print (usr_mov_clst)
#print(df_clusters)


# In[ ]:


all_clusters = []

t_results = helpers.scan(es, index = 'movies', query = {"query": {"match_all": {}}})
t_res_set = set()
for item in t_results:
    popo2 = item['_source']['movieId']
    t_res_set.add(int(popo2))

t_lst = sorted(t_res_set)

t_data = {'movieId': t_lst} 
df_mov_clst = pd.DataFrame.from_dict(t_data)
df_mov_clst = df_mov_clst.sort_values(by=['movieId'])

for i in range(10):
    all_clusters.append(i)

for i in range (len(all_clusters)) :
    df_mov_clst[all_clusters[i]] = pd.Series(None, dtype = 'float32')

#df_mov_clst


# In[ ]:


#constructing the dataframe takes some time
for t_movie in df_movies['movieId'].unique():
    #mean_list = []
    print ("movie : ", t_movie)
    for cluster in df_clusters['cluster'].unique():
        print ("cluster : ", cluster)
        mean = None
        mean = usr_mov_clst.loc[(usr_mov_clst['cluster'] == cluster) & (usr_mov_clst['movieId'] == t_movie), 'rating'].mean(axis = 0)
        #mean_list.append(mean)
        df_mov_clst.loc[df_mov_clst['movieId'] == t_movie, cluster] = mean
        print ("df", df_mov_clst.loc[df_mov_clst['movieId'] == t_movie, cluster]) 


# In[ ]:


#here we export the dataframe containing the average ratings for every movie per cluster to a .csv file
print("Exporting df_mov_clst_dataframe to csv ...")
df_mov_clst.to_csv (r'.\data\ratings_movies_clusters.csv', index = False, header=True)
print("Done!")

