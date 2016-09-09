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
import pickle
import numpy as np


LOGFORMAT = "%(asctime)s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen - Nukui Shun"
__email__ = "{hoangnt,nukui.s}@net.c.titech.ac.jp"

class Graph(defaultdict):

  def __init__(self, directed=False, name='graph'):
    """
    Parameters
    ----------
      name: Name string of the graph. (optional)
      directed: Directed or undirected. (optional)

    Returns
    -------
      none.
    """
    super(Graph, self).__init__(list)
    self._name = name
    self._directed = directed
    self._ids_list = None
    self._cur_idx = 0
    self._degrees = None
    self._distort = 1
    self._q = None
    self._J = None
    self._communities = None
    if directed:
      self._backward = list()

  def nodes(self):
    """
    Return list of nodes in graph.
    """
    if not hasattr(self, "_nodes"):
        self._nodes = list(self.keys())
    return self._nodes

  def random_walk(self, walk_length, start_node=None,
                  rand_seed=None, reset=0.0,
                  walk_bias=[0.99], isNeg=False):
    """
    Return a list of nodes in a truncated random walk.
    This function serves a comparision with our motif walk.
    Implementation similar to random walk of deepwalk model.

    Parameters
    ----------
      walk_length: Walk length is the number of random iteration
                   generated for each walk. The result doesn't
                   necessarily have all unique nodes.
      start_node: Node to start the random walk. Default is
                  none, in this case, the algorithm will select
                  a random node to be a start node. (Optional)
      rand_seed: An integer as random seed. If none, it will
                 use system time as seed. (Optional)
      reset: A float in [0.0,1.0] as reset to start node 
             probability.
      walk_bias: <unused>
      isNeg: Only return a single node if the function is
             used as negative sample generator.

    Returns
    -------
      walk_path: A list contains node ids of the walk.
    """
    assert 0 <= reset <= 1, 'Probability should be in [0.0,1.0].'
    random.seed(rand_seed)
    # Select starting node
    if start_node is None:
        start_node = random.choice(self.nodes())
    walk_path = np.ndarray(shape=(walk_length), dtype=np.int32)
    walk_path[0] = start_node
    cur = start_node
    for i in range(1, walk_length):
      if random.random() >= reset:
        walk_path[i] = random.choice(self[cur])
        cur = walk_path[i]
      else:
        walk_path[i] = walk_path[0]
        cur = walk_path[0]
    if isNeg:
      return random.choice(walk_path[(walk_length//2):])
    return walk_path

  def node2vec(self, walk_length, start_node=None,
               rand_seed=None, reset=.0,
               walk_bias=[0.25, 0.25], isNeg=False):
    """
    Implement biased walk in Node2Vec paper.

    Parameters
    ----------
    """

  def triangle_walk(self, walk_length, start_node=None, 
                    rand_seed=None, reset=0.0, 
                    walk_bias=[0.99], isNeg=False):
    """
    Walk follow the undirected triangle pattern.
  
    Parameters
    ----------
      walk_length: Length of the walk generated.
      start_node: Node to start the random walk. None means a random
                  node will be generated for starting the walk. (Optional)
      rand_seed: Random seed for random module. None means random
                 will use system time. (Optional)
      reset: <unused>
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
    random.seed(rand_seed)
    # Select starting node
    walk_path = np.ndarray(shape=(walk_length), dtype=np.int32)
    if start_node is None:
        start_node = random.choice(self.nodes())
    walk_path[0] = start_node
    cur = start_node
    prev = None
    for i in range(1, walk_length):
      cand = random.choice(self[cur])
      if prev:
        while True:
          prob = random.random()
          if cand in self[prev]:
            if prob < walk_bias[0]:
              walk_path[i] = cand
              break
          else:
            if prob > walk_bias[0]:
              walk_path[i] = cand
              break
          cand = random.choice(self[cur])
      else:
          walk_path[i] = cand
      prev = cur
      cur = cand
    if isNeg:
      return random.choice(walk_path[(walk_length//2):])
    return walk_path

  def unigram(self, walk_length=None, start_node=None, 
              rand_seed=None, reset=None, 
              walk_bias=[0.75], isNeg=True):
    """
    Special function to get random node in the graph.
    This function is designed to use as the negative 
    sampling method. No meaning in using this function
    as positive samples generator.

    Parameters
    ----------
      walk_length: <unused>
      start_node: <unused>
      rand_seed: Seed for python random.
      reset: <unused>
      walk_bias: Distortion parameter for unigram.
                 0 - Uniform random.
                 1 - Unigram.
                 0.75 - Recommended by cool people.
      isNeg: <unused>

    Returns
    -------
      A single node id samples from the distribution.
    """
    # Check if the unigram distribution existed
    if walk_bias[0] == self._distort:
      return alias_draw(self._J, self._q)
    self._distort = walk_bias[0]
    for i, j in enumerate(self._degrees):
      self._degrees[i] = j**walk_bias[0]
    self._J, self._q = alias_setup(self._degrees)
    return alias_draw(self._J, self._q)

  def gen_walk(self, pos_func, neg_func, pos_args, neg_args,
               walk_per_batch, walk_length, neg_samp, num_skip, 
               shuffle, window_size):
    """
    Generate data from positive and negative context generators.

    Parameters
    ----------
      pos_func: Name of positive sample generator
      neg_func: Name of negative sample generator
      pos_args: Dictionary contains parameters for pos_func
      neg_args: Dictionary contains parameters for neg_func
      walk_per_batch: Number of walk in this batch
      walk_length: Length of positive walk and negative walk
      neg_samp: Number of negative samples
      num_skip: Number of postitive samples (traditional naming)
      shuffle: Reset graph and shuffle node ids list.
      window_size: Window size for skipgram model.

    Returns
    -------
      A tuple of:
          A tuple of:
              targets: Target node id.
              classes: Class of that node id.
          labels: Labels (positive or negative sample)
          walk_per_batch: Book keeping variable.
    """
    if shuffle:
      self._ids_list = self.nodes()
      random.shuffle(self._ids_list)
      self._cur_idx = 0
    walk_per_batch = min(walk_per_batch,
                         (len(self._ids_list) - self._cur_idx))
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
    neg_args['isNeg'] = True 
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
            classi = random.randint(1, window_size-1)
          class_avoid.append(classi)
          classes[la] = buff[classi]
          labels[la] = 1.0
        for k in range(neg_samp):
          la = i * samples_per_walk + j * samples_per_node + num_skip + k
          neg_args['start_node'] = buff[0]
          neg_args['walk_length'] = walk_length
          targets[la] = buff[0]
          classes[la] = _neg_func(**neg_args)
          labels[la] = 0.0
    return ((targets, classes),labels, walk_per_batch) 

  def gen_training_community(self, la=0.5):
    print("\nERROR::Switched to use get_ids_labels.\n") 

  def get_ids_labels(self, split=0.5):
    """
    Generate training node ids and its community vector.

    Parameters
    ----------
      split: How many percent of the node will be used 
           for training.

    Returns
    -------
      train_ids: training ids
      train_labels : each label contain a list of comm
      test_ids: testing ids
      test_labels: testing labels
    """
    if self._communities is None:
      print("ERROR. Community not found.")
    if not self._ids_list:
      self._ids_list = self.nodes()
      random.shuffle(self._ids_list)
    labels = list()
    for i in self._ids_list:
      labels.append(self._communities[i])
    num_train = int(split * len(self.nodes()))
    return self._ids_list[:num_train], labels[:num_train], self._ids_list[num_train:], labels[num_train:]

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
  graph._degrees = np.ndarray(shape=(max(data.keys())+1))
  for key, val in data.items():
    graph[key] = val
    graph._degrees[key] = len(val)
  graph._distort = 1.0
  graph._J, graph._q = alias_setup(graph._degrees)
  if comm_filename != None:
    with open(comm_filename, 'rb') as pfile:
      graph._communities = pickle.load(pfile)
  return graph

# Alias method. https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
def alias_setup(probs):
  K = len(probs)
  q = np.zeros(K)
  J = np.zeros(K, dtype=np.int32)
  smaller = []
  larger = []
  for kk, prob in enumerate(probs):
    q[kk] = K * prob
    if q[kk] < 1.0:
      smaller.append(kk)
    else:
      larger.append(kk)
  while len(smaller) > 0 and len(larger) > 0:
    small = smaller.pop()
    large = larger.pop()
    J[small] = large
    q[large] = q[large] - (1.0 - q[small])
    if q[large] < 1.0:
      smaller.append(large)
    else:
      larger.append(large)
  return J, q

def alias_draw(J, q):
  K = len(J)
  kk = int(np.floor(np.random.rand()*K))
  if np.random.rand() < q[kk]:
    return kk
  else:
    return J[kk]
