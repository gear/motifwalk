"""Graph model and operations
"""
# Coding: utf-8
# Filename: graph.py
# Created: 2016-07-16
# Description:
# v0.0: File created
# v0.1: Random walk with node oriented
# v0.2: Add default motif walk (undirected triangle)
# motif walk has simple strategy of storing the prev node.
# v0.3: Generate set of nodes result from random and
# motif walk. Frequency of node is ignored.
# v0.4: Create a batch generator.
# v0.5: Create contrast walk generator.
# v0.6: Fix batch generator for new model (label={0,1}).
# v0.7: Switch back to simple generator.
# v0.8: Add community variable and label generator.
# v0.9: Clean up code and prepare for parallel.

# External modules
import random
import logging
import os
import time
import itertools
from itertools import chain
from threading import Thread
from collections import defaultdict, deque
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np


LOGFORMAT = "%(asctime)s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen - Nukui Shun"
__email__ = "{hoangnt,nukui.s}@net.c.titech.ac.jp"

ids_list = []
cur_idx = []

class Graph(defaultdict):
    """Graph is a dictionary contains nodes
    """
    def __init__(self, directed=False, name='graph'):
        """
        Create a graph as default_dict with default
        mapping to an empty list.

        Parameters
        ----------
          name: Name string of the graph. (optional)
          directed: Directed or undirected. (optional)

        Returns
        -------
          none.

        Effect
        ------
          Create a Graph object which is a default
          dictionary with default factor generate
          a dictionary mapping ids to node instances

        Examples
        --------
          citeseer = Graph()
          citeseer[20] = [1,3,4]
        """
        super(Graph, self).__init__(list)
        self._name = name
        self._directed = directed
        self._logger = None
        self._ids_list = None
        self._cur_idx = 0
        self._freq = None
        self._communities = None
        if directed:
          self._backward = list()


    def getLogger(self):
        """ 
        Create logger for the graph on demand.
        """
        if not self._logger:
            self._logger = logging.getLogger(self._name)
        return self._logger

    def nodes(self):
        """
        Return list of nodes in graph.
        """
        if not hasattr(self, "_nodes"):
            self._nodes = list(self.keys())
        return self._nodes

    def subgraph(self, node_list=[]):
        """
        Create and return a Graph instance as a subgraph
        of this Graph object.

        Parameters
        ----------
          node_list: list of nodes ids in the subgraph

        Returns
        -------
          subgraph: A copy of Graph contains only nodes
                    in node_ids list.
        """
        subgraph_name = self._name + '_sub_' + str(len(node_list))
        subgraph = Graph(directed=self._directed, name=subgraph_name)
        for node_id in node_list:
            if node_id in self:
                subgraph[node_id] = [
                    n for n in self[node_id] if n in node_list]
        return subgraph

    def volume(self, node_list=None):
        """
        Return volume (inner edges count) of the subgraph
        created by node_list.

        Parameters
        ----------
          node_list: list of nodes ids in the subgraph

        Returns
        -------
          volume: inner edges count of the subgraph
        """
        count = 0
        if node_list:
            subgraph = self.subgraph(node_list)
        else:
            subgraph = self
        if not subgraph._volume:
            for node in subgraph:
                count += len(subgraph[node])
            if subgraph._directed:
                subgraph._volume = count
                return count
            else:
                subgraph._volume = count // 2
                return count // 2
        else:
            return subgraph._volume

    def random_walk(self, length, start_node=None,
                    rand_seed=None, reset=0.0):
        """
        Return a list of nodes in a truncated random walk.
        This function serves a comparision with our motif walk.
        Implementation similar to random walk of deepwalk model.

        Parameters
        ----------
          length: Walk length is the number of random iteration
                  generated for each walk. The result doesn't
                  necessarily have all unique nodes.
          start_node: Node to start the random walk. Default is
                      none, in this case, the algorithm will select
                      a random node to be a start node. (Optional)
          rand_seed: An integer as random seed. If none, it will
                     use system time as seed. (Optional)
          reset: A float in [0.0,1.0] as reset to start node 
                 probability.

        Returns
        -------
          walk_path: A list contains node ids of the walk.
        """
        assert 0 <= reset <= 1, 'Restart probability should be in [0.0,1.0].'
        random.seed(rand_seed)
        if self._directed:
            self.getLogger().warn('Performing random walk on directed graph.')
        # Select starting node
        if not start_node:
            start_node = random.choice(self.nodes())
        walk_path = [start_node]
        # Start random walk by appending nodes to walk_path
        while len(walk_path) < length:
            cur = walk_path[-1]
            if len(self[cur]) > 0:
                if random.random() >= reset:
                    walk_path.append(random.choice(self[cur]))
                else:
                    walk_path.append(walk_path[0])
            else:
                break
        return walk_path

    # TODO: Generalize motif walk
    def motif_walk(self, length, start_node=None, rand_seed=None,
                   reset=0.0, walk_bias=0.8):
        """
        Walk follow the motif pattern. 

        Parameters
        ----------
          length: Length of the walk generated.
                start_node: Node to start the random walk. None means a random
                      node will be generated for starting the walk. (Optional)
          rand_seed: Random seed for random module. None means random
                     will use system time. (Optional)
          reset: Walk reset back to start node probability. 
          walk_bias: How strickly the walk will follow motif pattern.
                     Default value is 1.0 means the walk will always follow
                     the motif. This is value is how likely the walk is biased
                     toward the motif pattern. This parameter is implemented 
                     here as the rejection probability.
          motif: Motif object defines the walk pattern. None means
                 the walk will be in undirected triangle pattern for
                 undirected graph and bipartite pattern for directed
                 graph. (Optional)

        Returns
        -------
          walk_path: A list contains node ids of motif walk. This version of
                     simple motif walk uses list. The future version will use
                     set as data structure for the walk. 
        """
        # Now - Default as triangle motif (undirected).
        assert 0 <= reset <= 1, 'Restart probability should be in [0.0, 1.0].'
        random.seed(rand_seed)
        if self._directed:
            self.getLogger().warn('Performing motif walk on directed graph.')
        # Select starting node
        walk_path = []
        if not start_node:
            start_node = random.choice(self.nodes())
        walk_path.append(start_node)
        cur = start_node
        prev = None
        # Start random walk
        while len(walk_path) < length:
            # Uniformly choose adj candidate node at random
            cand = random.choice(self[cur])
            # If candidate is in previous adj node, select with prob=walk_bias
            if prev:
                while True:
                    prob = random.random()
                    if cand in self[prev]:
                        if prob < walk_bias:
                            walk_path.append(cand)
                            break
                    else:
                        if prob > walk_bias:
                            walk_path.append(cand)
                            break
                    cand = random.choice(self[cur])
            else:
                walk_path.append(cand)
            prev = cur
            cur = cand
        return walk_path

    def gen_walk(self, pos_func, neg_func, pos_args, neg_args,
                 walk_per_batch, walk_length, neg_samp, num_skip, 
                 shuffle=True, window_size):
        """
        Generate data from positive and negative context generators.
        """
        if shuffle:
            self._ids_list = self.nodes()
            random.shuffle(self._ids_list)
            self._cur_idx = 0
        walk_per_batch = min(walk_per_batch, (len(self._ids_list) - self._cur_idx))
        data_shape = walk_per_batch * (walk_length - window_size + 1) * (num_skip+neg_samp)
        targets = np.ndarray(shape=(data_shape), dtype=np.int32)
        classes = np.ndarray(shape=(data_shape), dtype=np.int32)
        labels = np.ndarray(shape=(data_shape), dtype=np.float32)
        idx = self._cur_idx
        self._cur_idx += walk_per_batch
        _pos_func = getattr(self, pos_func)
        _neg_func = getattr(self, neg_func)
        if pos_args is None:
            pos_args = dict()
        if neg_args is None:
            neg_args = dict()
        # A little messy computation for having better running time
        samples_per_node = num_skip + neg_samp
        samples_per_walk = (walk_length-window_size + 1) * samples_per_node
        for i in range(walk_per_batch):
            pos_args['start_node'] = self._ids_list[idx]
            pos_args['walk_length'] = walk_length
            pos_walk = _pos_func(**pos_args)
            idx += 1
            walk_index = 0
            buff = deque(maxlen=window_size)
            for _ in range(window_size-1):
                buff.append(pos_walk[walk_index])
                walk_index += 1
            for j in range(walk_length-window_size + 1):
                buff.append(pos_walk[walk_index+j])
                classi = 0
                class_avoid = [classi]
                for k in range(num_skip):
                    la = i * samples_per_walk + j * samples_per_node + k
                    targets[la] = buff[0]
                    while classi in class_avoid:
                        classi = random.randint(1, skip_window-1)
                    class_avoid.append(classi)
                    classes[la] = buff[classi]
                    labels[la] = 1.0
                for k in range(neg_samp):
                    la = i * samples_per_walk + j * samples_per_node + num_skip + k
                    targets[la] = buff[0]
                    classes[la] = random.choice(self.nodes())
                    labels[la] = 0.0
        return ((targets, classes),labels, walk_per_batch) 

    def gen_training_community(self, portion=0.1):
        """
        Generate list of data labels and corresponding node id.
        Guarantees to yeild all communities. Non-overlapping only.
        """
        if self._communities is None:
            print("ERROR. Community not found.")
        reverse_comm = defaultdict(list)
        ids = list()
        for key, val in self._communities.items():
            reverse_comm[val].append(key)
        for comm_id in reverse_comm.keys():
            num_true = int(portion*len(reverse_comm[comm_id]))
            cand = random.choice(reverse_comm[comm_id])
            for _ in range(num_true):
                while cand in ids:
                    cand = random.choice(reverse_comm[comm_id])
                ids.append(cand)
        labels = [self._communities[x] for x in ids] 
        combined = list(zip(ids,labels))
        random.shuffle(combined)
        return zip(*combined)

def graph_from_pickle(pickle_filename, comm_filename=None, **graph_config):
    """
    Load pickle file (stored as a dict or defaultdict) and
    return the graph as the graph object.

    Parameters
    ----------
      pickle_filename: File name in string of the desired picle file.
      graph_config: (keyword argument) This kwarg store configuration
                    for the newly created Graph object. Empty list implies
                    default Graph.

    Returns
    -------
      graph: Graph object with data from pickle file.
    """
    # Check if file exists. TODO: Use log and exit instead of assert.
    assert os.path.exists(
        pickle_filename), 'Pickle file not found. Please check.'
    with open(pickle_filename, 'rb') as pfile:
        data = pickle.load(pfile)
        # Make sure the data is in dictionary structure
        assert isinstance(
            data, dict), 'Graph data is not a dictionary. Please check.'
    # Create graph with configuration
    graph = Graph()
    if len(graph_config):
        for key, val in graph_config.items():
            setattr(graph, key, val)
    # Compute total frequency
    num_edges = 0
    for key, val in data.items():
        graph[key] = val
        num_edges += len(val)
    graph._freq = np.ndarray(shape=(num_edges))
    i = 0
    for key, val in data.items():
        for _ in range(len(val)):
            graph._freq[i] = key
            i += 1
    if comm_filename != None:
        with open(comm_filename, 'rb') as pfile:
            graph._communities = pickle.load(pfile)
    return graph
