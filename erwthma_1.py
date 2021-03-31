#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()


# In[ ]:


#1st
#ask from the user to insert the movie that will come out with the BM25 criterion
temp = 1
while (temp != 0):
    
    #Asking from the user to insert the movie
    movie = input("Insert the movie you want\n")
    res = es.search(index="movies", body={"query": {"match": {"title":"{}".format(movie)}}}, size = 1000)
    #to get only the value of total hits
    hits = res['hits']['total']['value']
    print("Total number of hits",hits)
    print("Num  Title  Score")
    i=0
    if hits!=0:
        for hit in res['hits']['hits']:
            i=i+1
            title = hit['_source'].get('title')
            score = hit['_score']
            
            print(i,"  ",title,"  ",score)
    else:
        print("No such result\n")
    
    temp = input("Do you want to search for another movie? (0 = No)\n")
    if (temp.isdigit()) : 
        temp = int(temp)
        if (int(temp) == 0) : print("Thank you! See you next time!")

