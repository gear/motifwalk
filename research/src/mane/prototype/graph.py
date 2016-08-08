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
# v1.0: Change graph architecture (include back-pointer)

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


# External modules
import random
import logging
import os
import time
import itertools
from itertools import chain
from threading import Thread
from collections import defaultdict
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np


LOGFORMAT = "%(asctime)s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

# >>> BEGIN CLASS 'graph' <<<
class Graph(defaultdict):
    """Graph is a dictionary contains nodes
    """
    # __init__

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
        self._edges = None
        self._logger = None
        self._volume = None
        self._freq = dict()

        global ids_list 
        global cur_idx
    # getLogger

    def getLogger(self):
        """ 
        Create logger for the graph on demand.
        """
        if not self._logger:
            self._logger = logging.getLogger(self._name)
        return self._logger

    # nodes
    def nodes(self):
        """
        Return list of nodes in graph.
        """
        if not hasattr(self, "_nodes"):
            self._nodes = self.keys()
        return self._nodes
    # edges
    # TODO: Implement edges

    def edges(self):
        """
        Return sets of edges tuples in graph.
        """
        return None
    # subgraph

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
    # volume

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
    # random_walk

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
        # TODO: Use log and exit instead of assert
        assert 0 <= reset <= 1, 'Restart probability should be in [0.0,1.0].'
        random.seed(rand_seed)
        # Give warning if graph is directed. TODO: Detail warning.
        # TODO: Add walk info.
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
    # motif_walk

    def motif_walk(self, length, start_node=None, rand_seed=None,
                   reset=0.0, walk_bias=0.9, motif=None):
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
        # TODO: Implement Motif class and delegate the walk to Motif
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
    # build_random_walk

    def build_random_walk(self, num_walk=20, length=128,
                          start_node=None, rand_seed=None, reset=0.0):
        """
        Perform random walk num_walk times to create a list and set of 
        random walk nodes. This method is usually used with fixed start node.

        Parameters
        ----------
          num_walk: Number of random walk. Default to be 20. (Optional)
          length: Length of each random walk. Default to be 128. (Optional)
          start_node: Start location for the random walk. None means a random
                      node will be selected. (Optional)
          rand_seed: Seed for random module. None means system time is used. (Optional)
          reset: Reset back to the start node probability for each random walk. 
                 Default 0.0. (Optional)

        Returns
        -------
          walk_path: List of visited nodes.
          set(walk_path): Set of unique nodes in the path.
        """
        if not start_node:
            self.getLogger().warn('Creating random walk set with random start node.')
        walk_path = []
        for _ in range(num_walk):
            rwp = self.random_walk(length=length, start_node=start_node,
                                   rand_seed=rand_seed, reset=reset)
            walk_path.extend(rwp)
        # TODO: Log the walk
        return set(walk_path)

    # build_motif_walk
    def build_motif_walk(self, num_walk=20, length=128,
                         start_node=None, rand_seed=None,
                         reset=0.0, walk_bias=0.9):
        """
        Perform motif walk num_walk times to create a list and set of 
        motif walk nodes. This method is usually used fixed start node.

        Parameters
        ----------
          num_walk: Number of motif walk. Default to be 20. (Optional)
          length: Length of each motif wal. Default to be 128. (Optional)
          start_node: Start location for the random walk. None means 
                      a random node will be selected. (Optional)
          rand_seed: Seed for random module. None means system time 
                     is used. (Optional)
          reset: Reset back to the start node probability for each 
                 motif walk. Default 0.0. (Optional)
        """
        if not start_node:
            self.getLogger().warn('Creating random walk set with random start node.')
        walk_path = []
        for _ in range(num_walk):
            mwp = self.motif_walk(length=length, start_node=start_node,
                                  rand_seed=rand_seed, reset=reset)
            walk_path.extend(mwp)
        return set(walk_path)

    ################################################################### gen_walk
    def gen_walk(self, walk_func_name, num_batches=100, walk_length=10,
                 num_walk=5, num_true=1, neg_samp=15,
                 num_skip=2, shuffle=True, window_size=3,
                 neg_samp_distort=0.75, gamma=0.8, n_threads=6, max_pool=10):
        """
        Generate walk sample in parallel
        """
        self._walk_pool = []
        self.max_pool = max_pool
        self.stop_threads = False
        self._threads = [Thread(target=self._gen_walk, name="gen_walk",
                   args=(walk_func_name, num_batches, walk_length,
                      num_walk, num_true, neg_samp, num_skip, shuffle,
                      window_size, gamma))
                   for _ in range(n_threads)]
        for t in self._threads:
            t.setDaemon(True)
            t.start()
        while True:
            if len(self._walk_pool) > 0:
                yield(self._walk_pool.pop())
            else:
                time.sleep(1)

    ################################################################## _gen_walk
    def _gen_walk(self, walk_func_name, num_batches, walk_length,
                  num_walk, num_true, neg_samp, num_skip, shuffle,
                  window_size, gamma):
    """
    Logic for gen_walk
    """


    def kill_threads(self):
        self.stop_threads = True

    def gen_contrast(self, possitive_name='motif_walk',
                      negative_name='random_walk', num_batches=100, reset=0.0,
                      walk_length=10, num_walk=5, num_true=1, neg_samp=15,
                      contrast_iter=10, num_skip=2, shuffle=True, window_size=3,
                      gamma=0.8, n_threads=8, max_pool=100):
        """
        Generate contrast walk in parallel
        """
        self._walk_pool = []
        self.max_pool = max_pool
        self._threads = [Thread(target=self._gen_contrast2, name="gen_contrast",
                   args=(possitive_name, negative_name, num_batches, reset,
                       walk_length, num_walk, num_true, neg_samp, contrast_iter,
                       num_skip, shuffle, window_size, gamma))
                   for _ in range(n_threads)]
        for t in self._threads:
            t.setDaemon(True)
            t.start()
        while True:
            if len(self._walk_pool) > 0:
                data = self._walk_pool.pop()
                yield data
            else:
                time.sleep(1)


    # gen_contrast
    def _gen_contrast(self, possitive_name='motif_walk',
                     negative_name='random_walk', num_batches=100, reset=0.0,
                     walk_length=10, num_walk=5, num_true=1, neg_samp=15,
                     contrast_iter=10, num_skip=2, shuffle=True, window_size=3,
                     gamma=0.8):
        """
        Create training dataset using possitive samples from motif walk
        and the negative samples from random walk.

        Parameters
        ----------
          possitive_name: Positive sample function name (e.g. 'motif_walk')
          negative_name: Negative sample function name (e.g. 'random_walk')
          num_batches: Number of batches per yield
          num_walk: Number of walk performed each starting node
          num_true: Number of true class for each sample.
          neg_samp: Number of negative sampling for each target.
          num_skip: Number of positive sampling for each target.
          shuffle: If node list is shuffled before generating random walk.
          window_size: Window for getting sample from the random walk list.
          gamma: Exponential decay for sampling distance

        Yields
        ------
          Yields a single tuple as the gen_walk function.
        """
        assert window_size >= num_skip, 'Window size is too small.'
        pos_func = getattr(self, possitive_name)
        neg_func = getattr(self, negative_name)
        # Generator loops forever
        while True:
            for _ in range(num_walk):
                if shuffle:
                    id_list = np.random.permutation(self.keys())
                else:
                    id_list = self.keys()
                # Accumulator for each batch
                count_batch = num_batches - 1
                targets = []
                classes = []
                labels = []
                weights = []
                eol = id_list[-1]
                for i in (id_list):
                    # Perform walk if the node is connected
                    if not len(self[i]) > 0:
                        continue
                    # Perform 2 walks and return set of nodes
                    pos_walk = []
                    neg_walk = []
                    for _ in range(contrast_iter):
                        pos_walk.extend(
                            pos_func(start_node=i, length=walk_length))
                        neg_walk.extend(
                            neg_func(start_node=i, length=walk_length))
                    # The set of negative samples is the contrast between 2
                    # walks
                    neg_samps_set = set(neg_walk) - set(pos_walk)
                    neg_samps = list(neg_samps_set)
                    if len(neg_samps) == 0:
                        print('Skipping empty set')
                        continue
                    pos_walk = pos_func(start_node=i, length=walk_length)
                    for j, target in enumerate(pos_walk):
                        # Window [lower:upper] for skipping
                        lower = max(0, j - window_size)
                        upper = min(walk_length, j + window_size + 1)
                        for _ in range(num_skip):
                            rand_index = random.randint(lower, upper-1)
                            rand_node = pos_walk[rand_index]
                            distance = abs(rand_index - j)
                            targets.append(target)
                            classes.append(rand_node)
                            labels.append(1.0)  # Possitive sample
                            # weight of positive sample
                            weights.append(pow(gamma, distance))
                        for _ in range(neg_samp):
                            rand_node = random.choice(neg_samps)
                            targets.append(target)
                            classes.append(rand_node)
                            labels.append(0.0)  # Negative sample
                            weights.append(1.0) # weight of neg. sample
                    if count_batch <= 0 or i == eol:
                        targets = np.array(targets, dtype=np.int32)
                        classes = np.array(classes, dtype=np.int32)
                        labels = np.array(labels, dtype=np.float32)
                        weights = np.array(weights, dtype=np.float32)
                        yield ({'target': targets, 'class': classes},
                                {'label': labels},
                                weights)
                        count_batch = num_batches - 1
                        targets = []
                        classes = []
                        labels = []
                        weights = []
                    else:
                        count_batch -= 1

# === END CLASS 'graph' ===


# >>> HELPER FUNCTIONS <<<

# graph_from_pickle
def graph_from_pickle(pickle_filename, **graph_config):
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
    # Load data to the graph
    for key, val in data.items():
        graph[key] = val
        graph._freq[key] = len(val)
    # TODO: Log result of graph creation
    return graph

# === END HELPER FUNCTIONS ===
