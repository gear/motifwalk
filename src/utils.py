import pickle as p
import networkx as nx

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
