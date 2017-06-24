import numpy as np
import re
import sys

from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from motifwalk.utils import timer
from motifwalk.model import Classifier

class TopKRanker(OneVsRestClassifier):
    """Python 3 and sklearn 0.18.1 compatible version
    of the original implementation.
    https://github.com/gear/deepwalk/blob/master/example_graphs/scoring.py"""
    def predict(self, emb, top_k_list, num_classes=39):
        """Predicts top k labels for each sample
        in the `features` list. `top_k_list` stores
        number of labels given in the dataset. This
        function returns a binary matrix containing
        the predictions."""
        assert emb.shape[0] == len(top_k_list)
        probs = np.asarray(super().predict_proba(emb))
        all_labels = np.zeros(shape=(emb.shape[0], num_classes))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for l in labels:
                all_labels[i][l] = 1.0
        return all_labels

def run_embedding_classify_f1(graph_all, emb_np_file, clf=LogisticRegression(),
                              test_ratio=[0.5], rand_state=2):
    """Run node classification for the learned embedding."""
    labels = graph_all.get_labels()
    if type(emb_np_file) is str:
        emb = np.load(emb_np_file)
    else:
        emb = emb_np_file
    averages = ["micro", "macro", "samples", "weighted"]
    for sr in splits_ratio:
        X_train, X_test, y_train, y_test = train_test_split(
            emb, labels, test_size=sr, random_state=run)
        top_k_list = get_top_k(y_test)
        mclf = TopKRanker(clf)
        mclf.fit(X_train, y_train)
        test_results = mclf.predict(X_test, top_k_list,
                                    num_classes=labels.shape[1])
        str_output = "Train ratio: {}\n".format(1.0 - sr)
        for avg in averages:
            str_output += avg + ': ' + str(f1_score(test_results, y_test,
                                                        average=avg)) + '\n'
            str_output += "Accuracy: " + \
                        str(accuracy_score(test_results, y_test)) + '\n'
            print(str_output)


# Helper function
def get_top_k(labels):
    """Return the number of classes for each row in the `labels`
    binary matrix. If `labels` is Linked List Matrix format, the number
    of labels is the length of each list, otherwise it is the number
    of non-zeros."""
    if isinstance(labels, csr_matrix):
        return [np.count_nonzero(i.toarray()) for i in labels]
    else:
        return [np.count_nonzero(i) for i in labels]
