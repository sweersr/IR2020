import pymongo
import numpy as np
from Preprocess import count_single
import spacy
from tqdm import tqdm

def Load_data():
    nlp = spacy.load("en_core_web_sm")

    client = pymongo.MongoClient()
    data = client.get_database('nordlys-v02')
    db = data.get_collection('dbpedia-2015-10')

    entries = []
    save_count = 0

    for i in tqdm(range(0, 4641890, 1000)):
        cursor = db.find({'<dbo:abstract>': {"$exists":True}},  {"_id": 1, '<dbo:abstract>':1}).skip(i).limit(1000)
        for k in cursor:
            entries.append(count_single(k, nlp))
        np.save(f'data_split/save_num{save_count}', entries)
        save_count += 1
        entries = []