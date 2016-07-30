"""Graph model and operations
"""
# Coding: utf-8
# Filename: graph.py
# Created: 2016-07-16
# Description:
## v0.0: File created
## v0.1: Random walk with node oriented
## v0.2: Add default motif walk (undirected triangle)
##       motif walk has simple strategy of storing the prev node.
## v0.3: Generate set of nodes result from random and 
##       motif walk. Frequency of node is ignored.
## v0.4: Create a batch generator.
## v0.5: Create contrast walk generator.

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# External modules
import logging
import os
import itertools
import multiprocessing
import cPickle as pickle
import numpy as np
from itertools import izip 
from time import time

# My modules
import motif
import util


LOGFORMAT = "%(asctime)s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

# >>> BEGIN CLASS 'graph' <<<
class Graph(dict):
  """Graph is a dictionary contains nodes
  """
  ##################################################################### __init__
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
    super(Graph, self).__init__()
    self._name = name
    self._directed = directed
    self._edges = None
    self._logger = None
    self._volume = None
    self._freq = dict()
  #################################################################### getLogger
  def getLogger(self):
    """ 
    Create logger for the graph on demand.
    """
    if not self._logger:
      self._logger = logging.getLogger(self._name)
    return self._logger

  ######################################################################## nodes
  def nodes(self):
    """
    Return list of nodes in graph.
    """
    return self.keys()
  ######################################################################## edges
  # TODO: Implement edges
  def edges(self):
    """
    Return sets of edges tuples in graph.
    """
    return None
  ##################################################################### subgraph
  def subgraph(self, node_list = []):
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
        subgraph[node_id] = [n for n in self[node_id] if n in node_list]
    return subgraph
  ####################################################################### volume
  def volume(self, node_list = None):
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
  ################################################################## random_walk
  def random_walk(self, length, start_node=None, 
                  rand_seed=None, reset = 0.0):
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
    rand = np.random
    rand.seed(rand_seed)
    # Give warning if graph is directed. TODO: Detail warning.
    # TODO: Add walk info.
    if self._directed:
      self.getLogger().warn('Performing random walk on directed graph.')
    # Select starting node
    if not start_node:
      start_node = rand.choice(self.keys())
    walk_path = [start_node]
    # Start random walk by appending nodes to walk_path
    while len(walk_path) < length:
      cur = walk_path[-1]
      if len(self[cur]) > 0:
        if rand.random() >= reset:
          walk_path.append(rand.choice(self[cur]))
        else:
          walk_path.append(walk_path[0])
      else:
        break
    return walk_path
  ################################################################### motif_walk
  def motif_walk(self, length, start_node=None, rand_seed=None, 
                 reset = 0.0, walk_bias = 0.9, motif=None):
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
    rand = np.random
    rand.seed(rand_seed)
    if self._directed:
      self.getLogger().warn('Performing motif walk on directed graph.')
    # Select starting node
    walk_path = [] 
    if not start_node:
      start_node = rand.choice(self.keys())
    walk_path.append(start_node)
    cur = start_node
    prev = None
    # Start random walk
    while len(walk_path) < length:
      # Uniformly choose adj candidate node at random
      cand = rand.choice(self[cur])
      # If candidate is in previous adj node, select with prob=walk_bias
      if prev:
        while True:
          prob = rand.random()
          if cand in self[prev]:
            if prob < walk_bias:
              walk_path.append(cand)
              break
          else:
            if prob > walk_bias:
              walk_path.append(cand)
              break
          cand = rand.choice(self[cur])
      else:
        walk_path.append(cand)
      prev = cur
      cur = cand
    return walk_path
  ############################################################ build_random_walk
  def build_random_walk(self, num_walk = 20, length=128, 
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
    for _ in xrange(num_walk):
      rwp = self.random_walk(length=length, start_node=start_node, 
                             rand_seed=rand_seed, reset=reset)
      walk_path.extend(rwp)
    # TODO: Log the walk
    return set(walk_path)

  ############################################################# build_motif_walk
  def build_motif_walk(self, num_walk = 20, length=128, 
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
    for _ in xrange(num_walk):
      mwp = self.motif_walk(length=length, start_node=start_node,
                            rand_seed=rand_seed, reset=reset)
      walk_path.extend(mwp)
    return set(walk_path)
  ############################################################ gen_with_negative
  def gen_walk(self, walk_func_name, walk_length=5, 
               num_walk=5, num_true=1, neg_samp=5, 
               num_skip=5, shuffle=True, window_size=5, 
               neg_samp_distort=0.75):
    """
    Infinite loop generating data as a simple skipgram model
    with negative sampling.
    
    Parameters
    ----------
      walk_func_name: Walk function name (e.g. 'random_walk')
      walk_length: Total number of nodes in each walk.
      num_walk: Number of walk performed for each starting node.
      num_true: Number of true class for each sample.
      neg_samp: Number of negative samples for each target.
      num_skip: Number of samples generated for each target.
      shuffle: If node list is shuffled before generating random walk.
      window_size: Window for getting sample from the random walk list.
      batch_size: Number of samples generated.
      neg_samp_distort: Distort the uniform distribution for negative 
                        sampling. Float value of 0.0 means uniform 
                        sampling and value of 1.0 means normal unigram 
                        sampling. This scheme is the same as in word2vec
                        model implementation.
  
    Yields
    ------
      Yields a single tuple: ( {'target':..., 'class':...}, {'label':...} )
        target: id of target node
        class: id of class associated with the target node
        label: 1 for possitive sample, -1 for negative sample.

    Note
    ----
      The number of samples for ... 
        - Each starting node: num_walk * walk_length * (num_skip + neg_samp) 
        - Each epoch: #nodes * num_walk * walk_length * (num_skip + neg_samp)
      E.g.
        - Default setting: 
            #nodes * 5 * 5 * (5 + 5) = 250 * #nodes
  
    """
    # Make sure generated dataset has correct count.
    assert window_size >= num_skip, 'Window size is too small.'
    wfunc = getattr(self, walk_func_name)
    # Node degree distribution distorted by neg_samp_distort
    node_list = self._freq.keys()
    freq_list = np.array(self._freq.values(), np.int32)**neg_samp_distort
    norm = sum(freq_list)
    freq_list = freq_list / norm
    # Infinite loop generating samples
    while True:
      if shuffle:
        id_list = np.random.permutation(self.keys())
      else:
        id_list = self.keys()
      for i in (id_list):
        # Perform walk if the node is connected
        if not len(self[i]) > 0:
          continue
        walk = []
        for _ in xrange(num_walk):
          walk.extend( wfunc(length=walk_length, start_node=i) )
        for j, target in enumerate(walk):
          # Window [lower:upper] for skipping
          lower = max(0, j - window_size)
          upper = min(len(walk), j + window_size+1)
          targets = []
          classes = []
          labels = []
          for _ in xrange(num_skip):
            rand_node = np.random.choice(walk[lower:upper])
            targets.append(target)
            classes.append(rand_node)
            labels.append(1.0) # Possitive sample
          for _ in xrange(neg_samp):
            rand_node = np.random.choice(node_list, p = freq_list)
            targets.append(target)
            classes.append(rand_node)
            labels.append(-1.0) # Negative sample
          targets = np.array(targets, dtype=np.int32)
          classes = np.array(classes, dtype=np.int32)
          labels = np.array(labels, dtype=np.float32)
          for t,c,l in izip(targets, classes, labels):
            yield ({'target':t, 'class':c},{'label':l})

  ################################################################# gen_contrast
  def gen_contrast(self, possitive_name='motif_walk', 
                   negative_name='random_walk',
                   num_walk=5, num_true=1, neg_samp=5,
                   num_skip=5, shuffle=True, window_size=5):
    """
    Create training dataset using possitive samples from motif walk
    and the negative samples from random walk.
    
    Parameters
    ----------
      possitive_name : Positive sample function name (e.g. 'motif_walk')
      negative_name : Negative sample function name (e.g. 'random_walk')
      num_walk: Number of walk performed each starting node
      num_true: Number of true class for each sample.
      neg_samp: Number of negative sampling for each target.
      num_skip: Number of positive sampling for each target.
      shuffle: If node list is shuffled before generating random walk.
      window_size: Window for getting sample from the random walk list.
  
    Yields
    ------
      Yields a single tuple as the gen_walk function.
    """
    assert window_size >= num_skip, 'Window size is too small.'
    pos_func = getattr(self, possitive_name)
    neg_func = getattr(self, negative_name)
    while True:
      if shuffle:
        id_list = np.random.permutation(self.keys())
      else:
        id_list = self.keys()
      for i in (id_list):
        # Perform walk if the node is connected
        if len(self[i]) > 0:
          count_nodes -= 1
        else:
          continue
        # Perform 2 walks and return set of nodes
        pos_walk = pos_func(start_node=i, length=walk_length, 
                            num_walk=num_walk) 
        neg_walk = neg_func(start_node=i, length=walk_length,
                            num_walk=num_walk)
        # The set of negative samples is the contrast between 2 walks
        neg_samps = neg_walk - pos_walk
        for j, target in enumerate(pos_walk):
          lower = max(0, j - window_size)
          upper = min(walk_length, j + window_size+1)
          for _ in xrange(num_skip):
            # TODO: Check situation where rand_node == target
            # TODO: Use num_true. Now assume default value
            rand_node = np.random.choice(walk[lower:upper])
            node_tuples.append([target, rand_node])
            labels.append(1.0) # Possitive sample
          for _ in xrange(neg_samp):
            rand_node = np.random.choice(node_list, p = freq_list)
            node_tuples.append([target, rand_node])
            labels.append(-1.0) # Negative sample
        if count_nodes <= 0:
          node_tuples = np.array(node_tuples, dtype=np.int32)
          labels = np.array(labels, dtype=np.float32)
          yield (node_tuples[:,0], 
                 node_tuples[:,1],
                 labels)
          count_nodes = nodes_in_batch
          labels = []
          node_tuples = []
    
# === END CLASS 'graph' ===


# >>> HELPER FUNCTIONS <<<

############################################################## graph_from_pickle
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
  assert os.path.exists(pickle_filename), 'Pickle file not found. Please check.'
  with open(pickle_filename, 'rb') as pfile:
    data = pickle.load(pfile)
    # Make sure the data is in dictionary structure
    assert isinstance(data, dict), 'Graph data is not a dictionary. Please check.'
  # Create graph with configuration
  graph = Graph()
  if len(graph_config): 
    for key, val in graph_config.iteritems():
      setattr(graph,key,val)
  # Load data to the graph
  for key, val in data.iteritems():
    graph[key] = val
    graph._freq[key] = len(val)
  # TODO: Log result of graph creation
  return graph

# === END HELPER FUNCTIONS ===
