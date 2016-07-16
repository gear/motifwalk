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

LOGFORMAT = "%(asctime)s %(levelname)s "
            "%(filename)s: %(lineno)s %(message)s"

__author__ = "Hoang Nguyen"
__email__ = "hoangnt@ai.cs.titech.ac.jp"

# >>> BEGIN CLASS 'node' <<<
class node(default_dict):

# === END CLASS 'node' ===

# >>> BEGIN CLASS 'graph' <<<
class Graph(default_dict):
  """Graph contains nodes stored as hash map
  """
  def __init__(self):
    """
    Create a graph as default_dict with default
    mapping to an empty list.
    
    Parameters
    ----------
      self: The object itself.

    Returns
    -------
      none.

    Effect
    ------
      Create a Graph object which is a default
      dictionary with default factor generate
      a dictionary mapping ids to node instances
    """
    super(Graph, self).__init__(dict)

  def nodes(self):
    return self.keys() 

  def iterator_nodes(self):
    return self.iteritems()

  def subgraph(self, node_ids = []):
    """
    Create and return a Graph instance as a subgraph
    of this Graph object.
    
    Parameters
    ----------
      node_ids: list of nodes ids in the subgraph

    Returns
    -------
      subgraph: A copy of Graph contains only nodes
                in node_ids list.
    """
    subgraph = Graph()
    for n in node_ids:
      if n in self:
        subgraph[n] = self.getnode(x)

  

# === END CLASS 'graph' ===



