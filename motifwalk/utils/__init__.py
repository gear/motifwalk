import pickle
import numpy as np
import re
import sys
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from time import time

dataloc = './../../data'
metasplit = r'%'
metadata = None

def find_meta(graph_name):
    """Find and return the metadata description of the given graph name.

    Parameters:
    graph_name - str - name of the dataset.

    Return:
    metum - str - metadata for the given graph name if found.

    Examples:
    meta_bc = find_meta('blogcatalog')
    "blogcatalog: blogcatalog.data
        graph: NXGraph
        labels: LILLabels
        convert_labels: False
        info: blogcatalog.md
        url:"
    """
    if metadata is None:
        print("Error: Metadata is not defined.")
        return None
    if not graph_name in all_graphs():
        print("Error: Graph '{}' is not found.".format(graph_name))
        return None
    for metum in metadata[1:]:
        if graph_name in metum:
            return metum
    print("Error: Graph '{}' is in header but not defined.".format(graph_name))
    sys.exit()

def all_graphs():
    return metadata[0].strip().split(':')[1].split()

def set_dataloc(path_to_data=None):
    global dataloc
    if path_to_data is not None:
        dataloc = path_to_data

def get_metadata():
    global metadata
    if metadata is None:
        with open(dataloc+'metadata') as f:
            metadata = f.read().split(metasplit)
    return metadata

def load_embeddings(emb_file):
    """Load graph embedding output from deepwalk, n2v to a numpy matrix."""
    with open(emb_file, 'rb') as efile:
        num_node, dim = map(int, efile.readline().split())
        emb_matrix = np.ndarray(shape=(num_node, dim), dtype=np.float32)
        for data in efile.readlines():
            node_id, *vector = data.split()
            node_id = int(node_id)
            emb_matrix[node_id, :] = np.array([i for i in map(np.float, vector)])
    return emb_matrix

class TopKRanker(OneVsRestClassifier):
    """Python 3 and sklearn 0.18.1 compatible version
    of the original implementation.
    https://github.com/gear/deepwalk/blob/master/example_graphs/scoring.py"""
    def predict(self, features, top_k_list, num_classes=39):
        """Predicts top k labels for each sample
        in the `features` list. `top_k_list` stores
        number of labels given in the dataset. This
        function returns a binary matrix containing
        the predictions."""
        assert features.shape[0] == len(top_k_list)
        probs = np.asarray(super().predict_proba(features))
        all_labels = np.zeros(shape=(features.shape[0], num_classes))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for l in labels:
                all_labels[i][l] = 1.0
        return all_labels
