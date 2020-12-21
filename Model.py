from Utils import read_json, combine_collection
import numpy as np
import os
import spacy
from tqdm import tqdm
from Preprocess import proces_query


def KL_div(query, doc_dir):
    scores = []
    count_dict = {}

    for key in query[2].keys():
        count_dict[key] = query[2][key] / query[1]

    doc_lst = os.listdir(doc_dir)

    for file in tqdm(doc_lst):
        fl = np.load(doc_dir + '/' + file, allow_pickle=True)
        for i, doc in enumerate(fl):
            score = 0

            for word in count_dict.keys():
                if word in doc[1].keys():
                    score += count_dict[word] * np.log(count_dict[word] / doc[1][word][1])
                else:
                    score += count_dict[word] * 100
            scores.append((doc[0], file, i, score))

    scores.sort(key=lambda x: x[3])
    np.save('KL_scores/' + query[0], scores[:1000])
    return scores[:1000]


def KL_all():
    nlp = spacy.load("en_core_web_sm")
    collection = read_json('collection.json')
    query_col = read_json('nordlys/data/dbpedia-entity-v2/queries_stopped.json')

    for key in query_col.keys():
        query = (key, query_col[key])
        print('processing ' + key + ': \n')
        query_proc = proces_query(query, nlp, collection)
        KL_div(query_proc, "data_split_JM")
        print('\n')

def create_new_col(score_file):
    file_dict = {}

    for i in score_file:
        if i[1] in file_dict.keys():
            file_dict[i[1]].append(i[2])
        else:
            file_dict[i[1]] = [i[2]]

    new_col = {}
    length = 0

    for key in tqdm(file_dict.keys()):

        file = np.load("reduced_data/" + key, allow_pickle=True)

        for index in file_dict[key]:
            length += file[int(index)][1]
            new_col = combine_collection(new_col, file[int(index)][2])

    new_col_LM = {}

    for key in new_col.keys():
        new_col_LM[key] = new_col[key] / length

    return new_col, new_col_LM, length, file_dict


def JM(word, doc, col):
    if word in doc[2].keys():
        return col[word] - doc[2][word] / doc[1], doc[2][word] / doc[1]
    else:
        return col[word], 0


def RM1(word, query, betas, doc_lst, collection):
    nume = np.zeros(len(betas))
    deno = np.zeros(len(betas))

    for key in doc_lst.keys():
        file = np.load("reduced_data/" + key, allow_pickle=True)

        for index in doc_lst[key]:
            num = np.ones(len(betas))
            doc = file[int(index)]

            for wrd in query:
                j1, j2 = JM(wrd, doc, collection)
                num *= (betas * j1 + j2)

            j1, j2 = JM(word, doc, collection)
            nume += num
            deno += num * (betas * j1 + j2)
    return deno / nume


def RM3(word, query, betas, doc_lst, collection, alphas):
    final = []
    RM = RM1(word, query[2], betas, doc_lst, collection)

    if word in query[2].keys():
        LM = query[2][word] / query[1]
    else:
        LM = 0

    for i in alphas:
        final.append(i * LM + (1 - i) * RM)

    return np.array(final)


def KL(collection, query, betas, doc_lst, alphas):
    scores = []
    RMs = []

    print("Calculating RMs: \n")
    for word in tqdm(collection.keys()):
        RMs.append(RM3(word, query, betas, doc_lst, collection, alphas))

    print("scoring all documents: \n")
    for key in tqdm(doc_lst.keys()):

        file = np.load("reduced_data/" + key, allow_pickle=True)

        for index in doc_lst[key]:

            score = 0

            doc = file[int(index)]

            for i, word in enumerate(collection.keys()):

                if word in doc[2].keys():
                    score += (RMs[i] * np.log(RMs[i] / doc[2][word] / doc[1]))
                else:
                    score += RMs[i] * 100
            scores.append((doc[0], score))
    return scores


def rerank_all(query_col, betas, alpha, nlp, clipping=None):
    for key in query_col.keys():
        if key + '.npy' in os.listdir("reranked_scores"):
            continue
        print("Processing " + key + ": \n")

        fle = np.load("KL_scores/" + key + '.npy', allow_pickle=True)[:100]

        print("Creating new collection")
        collection, collection_LM, _, file_dict = create_new_col(fle)

        if clipping != None:
            x = dict(sorted(collection.items(), key=lambda item: item[1], reverse=True))
            x = list(x.keys())[:clipping]
            for keys in list(collection.keys()):
                if keys in x:
                    continue
                else:
                    collection.pop(keys)
                    collection_LM.pop(keys)

        query = proces_query((key, query_col[key]), nlp, collection)

        np.save("reranked_scores/" + key, KL(collection_LM, query, betas, file_dict, alpha))

def rerank_top_100(file, alpha, beta):

    new_lst = []

    for item in file:
        new_lst.append((item[0], item[1][alpha, beta]))
    new_lst.sort(key=lambda x: x[1])
    return new_lst

def save_csv(folder, alpha, beta):

    df = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f'])
    folderlst = os.listdir(folder)
    for j, filename in tqdm(enumerate(folderlst)):
        temp_df = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f'])
        query_name = filename[:-4]
        reranked = np.load("reranked_scores/"+filename, allow_pickle=True)
        original = np.load("KL_scores/"+filename, allow_pickle=True)
        original = [(x[0], x[3]) for x in original]
        reranked = rerank_top_100(reranked, alpha, beta)
        original[:100] = reranked
        rank = []
        score = []
        for i in range(len(original)):
            original[i] = original[i][0]
            rank.append(i+1)
            score.append(1000 - i)

        temp_df['c'] = original
        temp_df['a'] = query_name
        temp_df['b'] = "Q0"
        temp_df['f'] = "RM3"
        temp_df['d'] = rank
        temp_df['e'] = score
        df = pd.concat([df, temp_df])
    df.to_csv(f'runs/rerank{alpha}{beta}.csv', index=False, header=False, sep='\t')