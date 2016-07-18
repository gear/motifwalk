"""Graph model and operations
"""
# Coding: utf-8
# Filename: graph.py
# Created: 2016-07-16
# Description:
## v0.0: File created
## v0.1: Random walk
## v0.2: Add default motif walk (undirected triangle)

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import random
import os
import itertools
import multiprocessing
import cPickle as pickle

import motif

from time import time

LOGFORMAT = "%(asctime)s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

# >>> BEGIN CLASS 'graph' <<<
class Graph(dict):
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
    super(Graph, self).__init__()
    self._name = name
    self._directed = directed
    self._edges = None
    self._logger = None

  def getLogger(self):
    """ Create logger for the graph on demand
    """
    if not self._logger:
      self._logger = logging.getLogger(self._name)
    return self._logger

  def nodes(self):
    """Return list of nodes in graph
    """
    return self.keys()

  # TODO: Implement edges
  def edges(self):
    """Return sets of edges tuples in graph
    """
    return None
    

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
  
  ### GRAPH - VOLUME ###
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
    subgraph = self.subgraph(node_list)
    count = 0
    for node in subgraph:
      count += len(subgraph[node])
    if self._directed:
      return count 
    else:
      return count // 2

  ### GRAPH - RANDOM_WALK ###
  def random_walk(self, length, start_node=None, rand_seed=None, reset = 0.0):
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
    rand = random.Random(rand_seed)
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

  ### GRAPH - MOTIF_WALK ###
  def motif_walk(self, length, motif=None, start_node=None,
                 rand_seed=None, reset = 0.0, walk_bias = 1.0):
    """
    Walk follow the motif pattern. 
    
    Parameters
    ----------
      length: Length of the walk generated.
      motif: Motif object defines the walk pattern. None means
             the walk will be in undirected triangle pattern for
             undirected graph and bipartite pattern for directed
             graph. (Optional)
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

    Returns
    -------
      walk_path: A set contains node ids of motif walk. Set data structure
                 is selected as it is useful to check membership in motif
                 walk. Also, the order of the walk is not every important
                 as later, subgraph is generated from the node collection
                 and samples will be picked from there. Negative will be
                 picked from substracting motif walk and random walk.
    """
    # TODO: Implement Motif class and delegate the walk to Motif
    # Now - Default as triangle motif (undirected)
    assert 0 <= reset <= 1, 'Restart probability should be in [0.0, 1.0].'
    rand = random.Random(rand_seed)
    if self._directed:
      self.getLogger().warn('Performing motif walk on directed graph.')
    # Select starting node
    walk_path = set()
    if not start_node:
      start_node = rand.choice(self.keys())
    walk_path.add(start_node)
    cur = start_node
    while _ in xrange(length):
      
      

# === END CLASS 'graph' ===

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
  # TODO: Log result of graph creation
  return graph
     
    






















































