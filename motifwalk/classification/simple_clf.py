import numpy as np
import re
import sys
import argparse

from scipy.sparse import csr_matrix

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from motifwalk.utils import timer
from motifwalk.utils import find_meta, set_dataloc, get_metadata, timer
from motifwalk.utils.Graph import GraphContainer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clf_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "Logistic Regression", "Logistic Regression CV"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    LogisticRegressionCV()]

def merge_row(nparr1, nparr2):
    return np.append(nparr1, nparr2, axis=1)

def average(nparr1, nparr2, r=(0.5,0.5)):
    return (r[0] * nparr1) + (r[1] * nparr2)

merge_types = ["merge", "average"]
merge_funcs = [merge_row, average]

def f1(predicted_labels, true_labels):
    averages = ["micro", "macro", "samples", "weighted"]
    scores = []
    for avg in averages:
        scores.append(f1_score(predicted_labels, true_labels, average=avg))
    return scores, averages

def accuracy(predicted_labels, true_labels):
    return [accuracy_score(predicted_labels, true_labels)], ["accuracy"]

def nmi(predicted_labels, true_labels):
    return [normalized_mutual_info_score(predicted_labels, true_labels)], ["NMI"]

metrics = ["f1", "accuracy", "nmi"]
metric_funcs = [f1, accuracy, nmi]

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--embedding_file", type=str,
                    help="Embedding file name WITHOUT extension '.emb.npy'.")
parser.add_argument("-ee", "--extra_embedding", type=str,
                    help="Auxilary embedding file name WITH ext.")
parser.add_argument("--merge_type", type=str,
                    help="If auxilary embedding exists, specify merge type.",
                    default="merge")
parser.add_argument("-d", "--dataset", type=str,
                    help="Dataset name (e.g. cora).")
parser.add_argument("--dloc", type=str,
                    help="Location of the data folder.",
                    default="/home/gear/Dropbox/CompletedProjects/motifwalk/data")
parser.add_argument("-c", "--classifier", type=str,
                    help="Type of one vs rest classifier will be used: " + \
                          ", ".join(clf_names), default="Logistic Regression")
parser.add_argument("--metric", type=str, help="Score metric to report.",
                    default="f1")
parser.add_argument("-tr", "--training_ratio", type=float,
                    help="Fraction of data used for training.",
                    default=0.5)
parser.add_argument("-rs", "--random_seed", type=int,
                    help="Random seed for train-test spliting function.",
                    default=0)


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

def main(_):
    args = parser.parse_args()

    print("Reading data ...")
    set_dataloc(args.dloc)
    metadata = get_metadata()
    graph = GraphContainer(find_meta(args.dataset), args.dloc)
    try:
        emb = np.load(args.embedding_file+".emb.npy")
    except FileNotFoundError:
        emb = np.load(args.embedding_file)
    try:
        with open(args.embedding_file+".info", 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("No info is found.")
    eemb = None
    if args.extra_embedding is not None:
        eemb = np.load(args.extra_embedding)
        merger = merge_funcs[merge_types.index(args.merge_type)]
        emb = merger(emb, eemb)
    labels = graph.get_labels()

    print("Fitting embedding to {} classifier ...".format(args.classifier))
    try:
        clf = classifiers[clf_names.index(args.classifier)]
    except ValueError:
        print("Error: {} is undefined.".format(args.classifier))
        sys.exit(0)
    X_train, X_test, y_train, y_test = train_test_split(
        emb, labels, train_size=args.training_ratio, random_state=args.random_seed)
    top_k_list = get_top_k(y_test)
    mclf = TopKRanker(clf)
    mclf.fit(X_train, y_train)
    test_results = mclf.predict(X_test, top_k_list,
                                num_classes=labels.shape[1])

    print("Reporting {} score for dataset {} with {} training ...".format(
                args.metric, args.dataset, args.training_ratio))
    sc_func = metric_funcs[metrics.index(args.metric)]
    sc, variation = sc_func(test_results, y_test)
    for s, v in zip(sc, variation):
        print("{} score: {}".format(v, s))

if __name__ == "__main__":
    main("main")
