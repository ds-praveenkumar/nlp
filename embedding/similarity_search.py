import os 
import pandas as pd
from sentence_transformers import SentenceTransformer,util

df = pd.read_csv('job_skills.csv')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

## create embedding corpus
corpus = df['Title']
corpus_embedding = embedder.encode( corpus, convert_to_tensor=True)

## run similarity
query = 'google cloud'
top_k = 10
query_embedding = embedder.encode(query,  convert_to_tensor=True)
hits = util.semantic_search( query_embedding, corpus_embedding, top_k=top_k)
hits = hits[0]

# show results
for hit in hits:
    hit_id = hit['corpus_id']
    article_data = df.iloc[hit_id]
    title = article_data['Title']
    print( '--', title , round(hit['score'], 2))

