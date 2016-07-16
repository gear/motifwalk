"""Graph model and operations
"""
# Coding: utf-8
# Filename: graph.py
# Created: 2016-07-16
# Description:
## v0.0: File created

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import random
import os
import itertools
import multiprocessing

import Motif

from time import time

LOGFORMAT = "%(asctime)s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

# >>> BEGIN CLASS 'node' <<<
class node(dict):

# === END CLASS 'node' ===

# >>> BEGIN CLASS 'graph' <<<
class Graph(_dict):
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
    self._logger = logging.getLogger(name)
    self._directed = directed
    self._edges = None

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
    subgraph = Graph(directed=self._directed)
    for node_id in node_list:
      if node_id in self:
        subgraph[node_id] = [n for n in self[node_id] if n in node_list]
    return subgraph
  
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
    if self.directed:
      return count // 2
    else:
      return count

# === END CLASS 'graph' ===
