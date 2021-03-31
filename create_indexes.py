#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#only need to run once

from elasticsearch import helpers, Elasticsearch
import csv

es = Elasticsearch()


# In[ ]:


#here we create the index of the movies in elasticsearch

#with helpers.bulk we create the index for the movies 
with open(r'.\data\movies.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    helpers.bulk(es, reader, index='movies', doc_type='my-type')


# In[ ]:


#here we create the index of the ratings in elasticsearch

#with helpers.bulk we create the index for the ratings 
with open(r'.\data\ratings.csv', encoding='utf-8') as g:
    reader = csv.DictReader(g)
    helpers.bulk(es, reader, index='ratings', doc_type='my-type')

