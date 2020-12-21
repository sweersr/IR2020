import json
from collections import Counter
from operator import add
from functools import reduce
import numpy as np
from tqdm import tqdm
import os

def read_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def combine_collection(item, collection):

    collection = dict(reduce(add, (Counter(dict(x)) for x in [collection, item])))

    return collection

def combine_file(filename):

    file = np.load(filename, allow_pickle=True)
    col = {}
    for i in file:
        col = combine_collection(col, i[2])

    return col

def combine_dir(dirname):

    os.chdir(dirname)
    dir_lst = os.listdir()
    col = {}

    for i in tqdm(dir_lst):
        col = combine_collection(col, combine_file(i))

    return col