import pickle
import numpy as np
import re
import sys
from sklearn.preprocessing import MultiLabelBinarizer

def strip(s):
    return s.strip()

class GraphContainer:

    def __init__(self, metadata, dataloc=''):
        """Setup graph object using metadata
        Parameters:
        metadata - list - contains metadata strings
        dataloc - str - root folder path
        """
        metadata = [d for d in map(strip, metadata) if len(d) > 0]
        self.graph_name, self.graph_file = map(strip, metadata[0].split(':'))
        for info in metadata[1:]:
            key, string = info.split(':')
            self.__dict__[key.strip()] = string.strip()
        self.dataloc = dataloc + self.graph_file

    def get_graph():
        with open(self.dataloc, 'rb') as f:
            data = pickle.load(f)
            return data[self.graph]

    def get_features():
        with open(self.dataloc, 'rb') as f:
            data = pickle.load(f)
            return data[self.features]

    def get_labels():
        with open(self.dataloc, 'rb') as f:
            data = pickle.load(f)
            labels = data[self.labels]
            if (self.convert_labels == 'True'):
                labels = MultiLabelBinarizer().fit_transform(
                            labels.reshape(labels.shape[0], 1))
            return labels
