# coding: utf-8

import util
from graph import create_graph
import pandas as pd


edge_file = "data/com-youtube.ungraph.txt"
com_file = "data/com-youtube.top5000.cmty.txt"


adj_list = util.txt_edgelist_to_adjlist(edge_file)
com_file = util.txt_community_to_dict_transposed(com_file)

# save to youtube_small_2.graph
g = create_graph("youtube_small_2", adj_list, com_file, remove_unlabeled=True)

num_edges = sum([len(a) for a in g.values()])//2

print("Create graph successfully")
print("Number of nodes: {}".format(len(g)))
print("Number of edges: {}".format(num_edges))
