import pickle as p
import numpy as np

dataloc = './../data/'

def load_citeseer():
    with open(dataloc+'citeseer.data', 'rb') as f:
        data = p.load(f)
        graph = data['NXGraph']
        features = data['CSRFeatures']
        labels = data['Labels']
    return graph, features, labels

def load_cora():
    with open(dataloc+'cora.data', 'rb') as f:
        data = p.load(f)
        graph = data['NXGraph']
        features = data['CSRFeatures']
        labels = data['Labels']
    return graph, features, labels

def load_blogcatalog():
    with open(dataloc+'blogcatalog.data', 'rb') as f:
        data = p.load(f)
        graph = data['NXGraph']
        features = None
        labels = data['LILLabels']
    return graph, features, labels

def load_embeddings(emb_file):
    """Load graph embedding output from deepwalk, n2v to a numpy matrix."""
    with open(emb_file, 'rb') as efile:
        num_node, dim = map(int, efile.readline().split())
        emb_matrix = np.ndarray(shape=(num_node, dim), dtype=np.float32)
        for data in efile.readlines():
            node_id, *vector = data.split()
            node_id = int(node_id)
            emb_matrix[node_id,:] = map(float, vector)
    return emb_matrix

"""https://github.com/gear/deepwalk/blob/master/example_graphs/scoring.py"""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import izip
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.