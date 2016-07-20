"""Helper function for various data preprocessing tasks.
"""
# Coding: utf-8
# Filename: util.py
# Created: 2017-07-18
# Description: 
## v0.0: File created.
##       Data packing from text file.
## v0.1: Add walk length statistic test.

from __future__ import division

import cPickle as pickle
import math
import os

def txt_edgelist_to_pickle(filename, pickle_name='default.graph'):
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
  assert os.path.exists(filename), 'File not found. Please check: %s' % (filename)
  # Read lines and create graph dict
  with open(filename, 'rb') as datatext:
    graph = dict()
    for textline in datatext.readlines():
      # Skip comments
      if textline[0] == '#': 
        continue
      edge = textline.split() # no arg -> white spaces
      assert len(edge) == 2, 'Invalid data line: %s' % textline
      try:
        t_edge = (int(edge[0]), int(edge[1]))
      except Exception:
        t_edge = (edge[0], edge[1])
      if graph.has_key(t_edge[0]):
        graph[t_edge[0]].append(t_edge[1])
      else:
        graph[t_edge[0]] = [t_edge[1]]
      if graph.has_key(t_edge[1]):
        graph[t_edge[1]].append(t_edge[0])
      else:
        graph[t_edge[1]] = [t_edge[0]] 
  print('Created graph with %d nodes' % len(graph))
  print('Pickling to file: %s' % pickle_name)
  # Save as a pickle file
  with open(pickle_name, 'wb') as pfile:
    pickle.dump(graph, pfile, pickle.HIGHEST_PROTOCOL)
  return graph

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
    walk_info.append((len(walk[0]),len(walk[1])))
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
