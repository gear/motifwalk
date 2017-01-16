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
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

class TopKRanker(OneVsRestClassifier):
    """Python 3 and sklearn 0.18.1 compatible version
    of the original implementation."""
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

