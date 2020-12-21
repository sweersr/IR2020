import numpy as np
import os
import spacy
import json
from tqdm import tqdm
from Utils import read_json

nlp = spacy.load("en_core_web_sm")

def tokenize(text, nlp):

    doc = nlp(text)

    tknzd = [token.lemma_ for token in doc if token.pos_ != "PUNCT"]

    return tknzd

def proces_query(query, nlp, col_dict):
    tknzd_query = tokenize(query[1], nlp)

    query_counts = {}

    chars = ['.', '-', '[', ']', '{', '}', '(', ')', '/', '\\', '=', '+', '_', '"', "'", ',', '?', '!', '@', '#', '$',
             '%', '^', '&', '*', '`', '~', ';', ':', '–']

    for token in tknzd_query:
        if token not in query_counts:
            query_counts[token] = 1
        else:
            query_counts[token] += 1

    for char in chars:
        for key in list(query_counts.keys()):
            if char in key:
                new_key = key.split(char)
                for i in new_key:
                    if i in col_dict.keys():
                        query_counts[i] = query_counts[key]
                query_counts.pop(key)

    for key in list(query_counts.keys()):
        if key not in col_dict:
            query_counts.pop(key)

    length = 0
    for i in query_counts.keys():
        length += query_counts[i]

    return (query[0], length, query_counts)

def count_single(item, nlp):
  item_dict = {}
  tknzd_item = tokenize(item['<dbo:abstract>'][0].lower(), nlp)

  for token in tknzd_item:

    if token not in list(item_dict.keys()):
      item_dict[token] = 1
    else:
      item_dict[token] += 1

  return (item['_id'], len(tknzd_item), item_dict)

def decrease_vocabulary():
    split_lst = ['.', '-', '[', ']', '{', '}', '(', ')', '/', '\\', '=', '+', '_', '"', "'", ',', '?', '!', '@', '#', '$',
                 '%', '^', '&', '*', '`', '~', ';', ':', '–']

    dict = read_json("collection.json")

    for key in list(dict.keys()):
        if dict[key] == 1:
            dict.pop(key)

    dict = json.dumps(dict)
    f = open("collection.json", "w")
    f.write(dict)
    f.close()

    for char in split_lst:
        dict = read_json("collection.json")
        print(len(dict))
        keys = list(dict.keys())
        for key in tqdm(keys):
            if char in key:
                splits = key.split(char)
                for i in splits:
                    try:
                        dict[i] += dict[key]
                    except:
                        pass
                dict.pop(key)
        dict = json.dumps(dict)
        f = open("collection.json", "w")
        f.write(dict)
        f.close()

    dict = read_json("collection.json")

    file_lst = os.listdir('data_split')

    for file in tqdm(file_lst):
        data = np.load('data_split/' + file, allow_pickle=True)
        for j, point in enumerate(data):
            for char in split_lst:
                doc_dict = point[2]
                doc_keys = list(doc_dict.keys())
                for key in doc_keys:
                    if char in key:
                        splits = key.split(char)
                        for i in splits:
                            if i in doc_keys:
                                try:
                                    doc_dict[i] += doc_dict[key]
                                except:
                                    doc_dict[i] = doc_dict[key]
                        doc_dict.pop(key)
                point[2] = doc_dict
                data[j] = point
        np.save('reduced_data/' + file, data)

    for file in tqdm(file_lst):
        data = np.load('reduced_data/' + file, allow_pickle=True)
        for j, point in enumerate(data):
            doc_dict = point[2]
            doc_keys = list(doc_dict.keys())
            for key in doc_keys:
                if key not in dict.keys():
                    doc_dict.pop(key)
            point[2] = doc_dict
            data[j] = point
        np.save('reduced_data/' + file, data)

    for file in tqdm(file_lst):
        data = np.load('reduced_data/' + file, allow_pickle=True)
        for j, point in enumerate(data):
            doc_count = 0
            doc_dict = point[2]
            doc_keys = list(doc_dict.keys())
            for key in doc_keys:
                doc_count += doc_dict[key]
            point[1] = doc_count
            data[j] = point
        np.save('reduced_data/' + file, data)