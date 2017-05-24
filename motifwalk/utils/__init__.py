import pickle
import numpy as np
import re
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

def set_dataloc(path_to_data=None):
    global dataloc
    if path_to_data is not None:
        dataloc = path_to_data

# Metadata stores data reading instructions
with open(dataloc+"/metadata") as f:
    metadata = f.read()
