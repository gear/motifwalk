import pickle
import numpy as np
import re
import sys
import networkx as nx
import graph_tool as gt
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
        metadata = [d for d in map(strip, metadata.split('\n'))
                    if len(d) > 0]
        self.graph_name, self.graph_file = map(strip,
                                               metadata[0].split(':'))
        for info in metadata[1:]:
            key, string = info.split(':')
            self.__dict__[key.strip()] = string.strip()
        self.dataloc = dataloc + '/'

    def get_graph(self):
        with open(self.dataloc+self.graph_file, 'rb') as f:
            data = pickle.load(f)
            return data[self.graph]

    def get_gt_graph(self):
        nx_graph = self.get_graph()
        gt_graph = gt.Graph()
        gt_graph.set_directed(nx_graph.is_directed())
        gt_graph.add_edge_list(nx_graph.edges())
        return gt_graph

    def get_features(self):
        try:
            features_kw = self.features
        except AttributeError:
            print("{} doesn't have features.".format(self.graph_name))
            return None
        with open(self.dataloc+self.graph_file, 'rb') as f:
            data = pickle.load(f)
            return data[features_kw]

    def get_labels(self):
        try:
            labels_kw = self.labels
        except AttributeError:
            print("{} doesn't have labels.".format(self.graph_name))
            return None
        with open(self.dataloc+self.graph_file, 'rb') as f:
            data = pickle.load(f)
            labels = data[labels_kw]
            if (self.convert_labels == 'True'):
                labels = MultiLabelBinarizer().fit_transform(
                            labels.reshape(labels.shape[0], 1))
            return labels

    def write_gml(self):
        nx_graph = self.get_graph()
        nx_graph.name = self.graph_name
        nx.write_gml(nx_graph, self.dataloc+self.graph_name+'.gml')
