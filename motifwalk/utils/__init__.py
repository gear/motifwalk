import pickle
import numpy as np
import re
import sys
from time import time

dataloc = './../../data'
metasplit = r'%'
metadata = None

global start

def timer(begin=True):
    global start
    now = time()
    if begin:
        start = time()
    else:
        delta = now - start
        print("Time elapsed: {} sec".format(delta))

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
        raise ValueError("Metadata is not defined.")
    if not graph_name in all_graphs():
        raise ValueError("Error: Graph '{}' is \
                          not found.".format(graph_name))
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
        with open(dataloc+'/metadata') as f:
            metadata = f.read().split(metasplit)
    return metadata
