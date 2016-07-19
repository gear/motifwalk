"""Helper function for various data preprocessing tasks.
"""
# Coding: utf-8
# Filename: util.py
# Created: 2017-07-18
# Description: 
## v0.0: File created.
##       Data packing from text file.

import cPickle as pickle
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
  datatext = open(filename, 'rb')
  graph = dict()
  for textline in datatext.readlines():
    edge = textline.split(' ')
    assert len(edge) == 2, 'Invalid data line.'
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
  pfile = open(pickle_name, 'wb')
  pickle.dump(graph, pfile, pickle.HIGHEST_PROTOCOL)
  return graph
