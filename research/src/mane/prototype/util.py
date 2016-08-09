"""Helper function for various data preprocessing tasks.
"""
# Coding: utf-8
# Filename: util.py
# Created: 2017-07-18
# Description:
# v0.0: File created.
# Data packing from text file.
# v0.1: Add walk length statistic test.

from __future__ import division

try:
    import cPickle as pickle
except:
    import pickle
from itertools import chain
import math
import os
import pandas as pd
import csv


def txt_edgelist_to_adjlist(filename, pickle_name=None, split=None):
    """ 
    Sparse text file for edgelist and store 
    them as dictionary pickle file. 
    For undirected edge list file with non 
    repeating edges.

    Parameters
    ----------
      filename: path to the text file

    Returns
    -------
      graph: graph adjlist dictionary

    """
    assert os.path.exists(
        filename), 'File not found. Please check: %s' % (filename)
    # Read lines and create graph dict
    with open(filename, 'r') as datatext:
        graph = dict()
        for textline in datatext.readlines():
            # Skip comments
            if textline[0] == '#':
                continue
            if split is not None:
              textline = textline.strip()
            edge = textline.split(split)  # no arg -> white spaces
            assert len(edge) == 2, 'Invalid data line: %s' % textline
            try:
                t_edge = (int(edge[0]), int(edge[1]))
            except Exception:
                t_edge = (edge[0], edge[1])
            if t_edge[0] in graph:
                graph[t_edge[0]].append(t_edge[1])
            else:
                graph[t_edge[0]] = [t_edge[1]]
            if t_edge[1] in graph:
                graph[t_edge[1]].append(t_edge[0])
            else:
                graph[t_edge[1]] = [t_edge[0]]
    print('Created graph with %d nodes' % len(graph))
    print('Pickling to file: %s' % pickle_name)
    if pickle_name:
        # Save as a pickle file
        with open(pickle_name, 'wb') as pfile:
            pickle.dump(graph, pfile, pickle.HIGHEST_PROTOCOL)


def walk_length_stat_test(build_walk_func, num_walk,
                          walk_length, start_node, num_build):
    """
    Perform multiple walk corpus build using build_walk_func
    and report mean and deriviation of motif walk and random
    walk.

    Parameters
    ----------
      build_walk_func: function to generate walks. This function
                       must belong to an object (see Example).
      num_walk: number of walk per build.
      walk_length: length of each walk.
      start_node: starting node for the build.
      num_build: number of times build_walk_func is ran for stat
                 test.

    Returns
    -------
      None. Function print out mean and sample deriviation.
      Format of the print:
      (mean_unique_node_count, deriviation_unique_node,
       mean_all_path_node_count, deriviation_path_node)

    Examples
    --------
      >>> fbgraph = graph.graph_from_pickle('data/egonets.graph')
      >>> 
      >>> util.walk_length_stat_test(fbgraph.build_motif_walk,
      >>>                            num_walk=20,
      >>>                            length=128,
      >>>                            start_node=1,
      >>>                            num_buld=1000)
      ...
      >>> (515.174, 96.7813533401675, 2560.0, 50.6217)
    """
    count_unique = 0
    count_path = 0
    walk_info = []
    for _ in xrange(num_build):
        walk = build_walk_func(num_walk, length=walk_length,
                               start_node=start_node)
        walk_info.append((len(walk[0]), len(walk[1])))
        count_unique += len(walk[1])
        count_path += len(walk[0])
    avg_unique = count_unique / num_build
    avg_path = count_path / num_build
    count_unique = 0
    count_path = 0
    for i, j in walk_info:
        count_path += (i - avg_path)**2
        count_unique += (j - avg_unique)**2
    # Sample variance
    var_path = count_path / (num_build - 1)
    var_unique = count_unique / (num_build - 1)
    return avg_unique, math.sqrt(var_unique), avg_path, math.sqrt(var_path)



def txt_community_to_dict_transposed(filename, picklename=None):
    """
    File format: line[node_list] (line no. corresponds to com_id)

    Input file format = space separated
    """
    com_cnt = dict()
    community = dict()
    with open(filename, 'r') as txt:
        for i, line in enumerate(txt.readlines()):
            nodes = list(map(int, line.strip().split()))
            com_cnt[i] = len(nodes)
            community[i] = set(nodes)
    sorted_coms = [c for c, _ in sorted(com_cnt.items(), key=lambda x: x[1])]
    # larger community is prior for overlapping node coms
    node_to_com = {}
    for c in sorted_coms:
        for n in community[c]:
            node_to_com[n] = c
    if picklename:
        with open(picklename, 'wb') as pfile:
            pickle.dump(node_to_com, pfile)
    return node_to_com


def txt_community_to_dict(filename, picklename=None):
    """
    File format: line[node_id community_id]
    
    Input file format = space separated, columns=[node_id, com_id]):
    """
    community = dict()
    com_cnt = dict()
    with open(filename, 'rb') as txt:
        for line in txt.readlines():
            arr = line.split()
            if not int(arr[1]) in community:
                community[int(arr[1])] = set()
            community[int(arr[1])].add(int(arr[0]))
            com_cnt[int(arr[1])] = com_cnt.get(int(arr[1]), 0) + 1
    sorted_coms = [c for c, _ in sorted(com_cnt.items(), key=lambda x: x[1])]
    # larger community is prior for overlapping node coms
    node_to_com = {}
    for c in sorted_coms:
        for n in community[c]:
            node_to_com[n] = c
    if picklename:
        with open(picklename, 'wb') as pfile:
            pickle.dump(node_to_com, pfile)
    return node_to_com


def txt_adjlist_to_pickle(filename, picklename='default.graph'):
    """
    Create pickle graph file from adjlist
    """
    graph = dict()
    with open(filename, 'rb') as txt:
        for line in gfile.readlines():
            arr = line.split()
            for i in arr[1:]:
                if not int(arr[0]) in graph:
                    graph[int(arr[0])] = list()
                if int(i) not in graph[int(arr[0])]:
                    graph[int(arr[0])].append(int(i))
    with open(picklename, 'wb') as pfile:
        pickle.dump(graph, pfile, pickle.HIGHEST_PROTOCOL)


def pickle_edges_to_txt(pickle_path, txt_path):
    """
    Read pickle and write pickle of edge list
    (Used in python3)
    """
    edges = pd.read_pickle(pickle_path)
    with open(txt_path, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(edges)


def pickle_labels_to_txt(pickle_path, txt_path):
    """
    Read pickle and write pickle of labels list
    (Used in python3)
    """
    labels = enumerate(pd.read_pickle(pickle_path))
    with open(txt_path, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(labels)


def read_correct_labels(txt_file, index_columns=True):
    labels = {}
    with open(txt_file) as f:
        for i, line in enumerate(f):
            data = line.strip().split(" ")
            if index_columns:
                labels[int(data[0])] = int(data[1])
            else:
                labels[i] = int(data[0])
    labels = [labels[i] for i in range(len(labels))]
    return labels

def reindex_edges_and_community(adj_list, com_dict=None):
    unique_set = set(adj_list.keys())
    unique_set |= set(chain.from_iterable(adj_list.values()))
    zip_ids = {nid: ind for ind, nid in enumerate(unique_set)}
    new_adj_list = {}
    for k, adj in adj_list.items():
        new_adj_list[zip_ids[k]] = [zip_ids[a] for a in adj]
    if com_dict:
        new_com_dict = {zip_ids[i]: c for i, c in com_dict.items()}
    else:
        return new_adj_list
    return new_adj_list, new_com_dict
