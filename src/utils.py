import pickle as p
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

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


def load_data(dataset_name):
    """Load dataset"""
    if dataset_name == "blogcatalog":
        return load_blogcatalog()
    elif dataset_name == "cora":
        return load_cora()
    elif dataset_name == "citeseer":
        return load_citeseer()
    else:
        raise ValueError("Dataset not found")


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


def get_top_k(labels):
    """Return the number of classes for each row in the `labels`
    binary matrix. If `labels` is Linked List Matrix format, the number
    of labels is the length of each list, otherwise it is the number
    of non-zeros."""
    if isinstance(labels, lil_matrix):
        return [len(i) for i in labels]
    else:
        return [np.count_nonzero(i.toarray()) for i in labels]


def run_embedding_classify_f1(dataset_name, emb_file, clf=LogisticRegression(),
                           splits_ratio=[0.5], num_run=2, write_to_file=None):
    """Run node classification for the learned embedding."""
    _, _, labels = load_data(dataset_name)
    emb = load_embeddings(emb_file)
    results_str = []
    averages = ["micro", "macro", "samples", "weighted"]
    for run in range(num_run):
        results_str.append("\nRun number {}:\n".format(run+1))
        for sr in splits_ratio:
            X_train, X_test, y_train, y_test = train_test_split(
                emb, labels, test_size=sr, random_state=run)
            top_k_list = get_top_k(y_test)
            mclf = TopKRanker(clf)
            mclf.fit(X_train, y_train)
            test_results = mclf.predict(X_test, top_k_list,
                                        num_classes=labels.shape[1])
            str_output = "Train ratio: {}\n".format(sr)
            for avg in averages:
                str_output += avg + ': ' + str(f1_score(test_results, y_test,
                                                    average=avg)) + '\n'
            results_str.append(str_output)
    info = "Embedding dim: {}, graph: {}".format(emb.shape[1], dataset_name)
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(info)
    print(''.join(results_str))


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

