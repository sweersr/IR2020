from Load_data import Load_data
from Utils import combine_dir, read_json
from Preprocess import decrease_vocabulary
from Model import KL_all, rerank_all, save_csv
import numpy as np
import json
import spacy

# First we use the Load_data function to load the data from our database, do some preprocessing and save the results
# to intermediate files in the folder data_split

Load_data()

# Use the newly loaded data to create and save a collection dictionary

collection = combine_dir('data_split')

collection = json.dumps(collection)
f = open("collection.json", "w")
f.write(collection)
f.close()

# decrease the corpus size of both the entire collection and of each document save intermediate results in reduced_data
decrease_vocabulary()

# perform KL-divergence on the data and save results in KL_scores
KL_all()

# finally perform the re-ranking using RM3 and save the results to runs
query_col = read_json('nordlys/data/dbpedia-entity-v2/queries_stopped.json')
nlp = spacy.load("en_core_web_sm")

rerank_all(query_col, np.arange(1, 10) / 10, np.arange(1, 10) / 10, nlp, 1000)

for i in range(9):
    for j in range(9):
        print(i, j)
        save_csv("reranked_scores", i, j)