from label_processor import *
from build_models import *
from feature_extractor import *
import cv2
import numpy as np
import math
import pandas as pd
from tqdm.auto import tqdm
import random
import string
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
import timm
import torch
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device} for inference')

queries = pd.read_csv('data/queries.csv')
test = pd.read_csv('data/test.csv')


class LabelInfo:
    def __init__(self):
        self.ids = []
        self.extra_embs = []
        self.embs = []
        self.mean_emb = None
        self.neigh = None

    def add(self, idx, emb):
        self.ids.append(idx)
        self.embs.append(emb)

    def process(self):
        for i in range(len(self.embs)):
            self.embs[i] = self.embs[i][None, :]
        self.embs = np.concatenate(self.embs, axis=0)
        self.mean_emb = np.mean(self.embs, axis=0)

        self.neigh = NearestNeighbors(n_neighbors=min(10, self.embs.shape[0]), metric='cosine')
        self.neigh.fit(self.embs)

        return self.mean_emb

    def get_best_ids(self, emb, k, mean_distance):
        if not k:
            return np.array([]), np.array([], dtype=int)
        k = min(k, self.embs.shape[0])
        distances, idxs = self.neigh.kneighbors(emb[None, :], k, return_distance=True)
        distances = 1 - distances[0]
        idxs = idxs[0]
        for i in range(k):
            idxs[i] = self.ids[idxs[i]]
            distances[i] = (1 * distances[i] + 100000 * mean_distance) / 100001
        return distances, idxs


def get_result(db, que, models_data_, db_dir='data/test/', que_dir='data/queries/'):

    # db = db.iloc[:10]
    # que = que.iloc[:5]

    label2id = {}
    id2label = []

    label_info = []

    db_emb = get_emb(db, models_data_, db_dir)
    que_emb = get_emb(que, models_data_, que_dir)

    for (_, row), emb in zip(db.iterrows(), db_emb):
        idx = row.idx
        label = row.item_nm
        if db.shape[0] > 10:
            label = label_process(label)

        if label not in label2id:
            label2id[label] = len(id2label)
            id2label.append(label)
            label_info.append(LabelInfo())
        label_info[label2id[label]].add(idx, emb)

    general_emb = []
    for i in range(len(label_info)):
        general_emb.append(label_info[i].process())
    general_emb = np.array(general_emb)

    general_neigh = NearestNeighbors(n_neighbors=10, metric='cosine')
    general_neigh.fit(general_emb)
    mean_distance, folder_ids = general_neigh.kneighbors(que_emb, 10, return_distance=True)
    mean_distance = 1 - mean_distance

    result = []
    for i, que_idx in enumerate(que.idx):
        ids = folder_ids[i]
        distances = np.array([])
        db_ids = np.array([], dtype=int)
        for j in range(10):
            label_id = ids[j]
            res = label_info[label_id].get_best_ids(que_emb[i], 10, mean_distance[i][j])
            distances = np.concatenate([distances, res[0]])
            db_ids = np.concatenate([db_ids, res[1]])

        best_scores = np.argsort(distances)[-10:]
        distances = distances[best_scores]
        db_ids = db_ids[best_scores]

        for j in range(10):
            result.append((que_idx, db_ids[j], distances[j]))

    assert(len(result) == 10 * que.shape[0])
    return result


result = get_result(test, queries, models_data, 'data/test/', 'data/queries/')

pred_data = pd.DataFrame()
pred_data['score'] = [row[2] for row in result]
pred_data['database_idx'] = [row[1] for row in result]
pred_data.loc[:, 'query_idx'] = [row[0] for row in result]

pred_data.to_csv('data/submission.csv', index=False)
