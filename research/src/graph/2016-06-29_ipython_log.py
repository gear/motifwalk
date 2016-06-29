# coding: utf-8
get_ipython().magic(u'clear ')
import networkx as nx
G = nx.graph
G
del G
G = nx.Graph()
G.add_edge(1,2)
G.add_edge(1,3)
G.add_edges_from.__format__
G.add_edges_from.__doc__
elist = [(1,4),(1,5), (1,6), (1,7)]
G.add_edges_from(elist)
G.edge
G.edges
G.edges()
type(G.edge)
G.edge(1)
G.edge[1]
G.edge[1][1]
G.edge[1][2]
G.edge[1][3]
G.edge[1][2] = {1: 'the'}
G.edges
G.edges()
G.edge(1,2)
G.edge[1,2]
G.edge[1][2]
G.edge[1][2][1]
elist = [('a', 'b', 4.0), ('b', 'c', 3.0)]
G.add_edges_from(elist)
G.add_weighted_edges_from(elist)
G.edges()
G.edge_attr_dict_factory()
G.edges_iter
G.edges_iter()
for e in G.edges_iter():
    print(e)
for e in G.edges_iter():
    print(G[e])
G[1]
G['a']
G.a.b.weight
G.add_edge(1,3)
G.add_edge((1,3))
G.add_edge(1,3)
G.neighbors
G.neighbors()
G.neighbors(1)
G[1]
nx.triangles(G,1)
get_ipython().magic(u'clear ')
G=nx.Graph()
e=[('a','b',0.3), ('b','c',0.9), ('a','c',0.5), ('c','d',1.2)]
G.add_weighted_edges_from(e)
print(nx.dijkstra_path(G, 'a', 'd'))
G = nx.cubical_graph()
nx.draw(G)
G=nx.Graph()
G.add_edge('A', 'B')
G.add_edge('B', 'C')
print(G.adj)
G.adj
G
G.edge
G.adj == G.edge
G.add_edge('A', 'D', color='blue', weight=0.84, size=300) 
G.edge
G
G[A][D]['color']
G['A']['D']['color']
get_ipython().magic(u'ls ')
import os
os.chdir('~/Dropbox/WorkingFiles/motifwalk/research/src/graph/')
os.chdir('~/Dropbox/WorkingFiles/motifwalk/research/src/graph')
os.chdir('~/Dropbox/WorkingFiles/motifwalk/research/src/graph')
os.chdir('/home/hoangnt/Dropbox/WorkingFiles/motifwalk/research/src/graph/')
facebook_graph = open('./data/facebook_combined.txt')
fg = nx.Graph(facebook_graph)
fg = nx.Graph()
fg = nx.read_adjlist(facebook_graph)
len(fg.neighbors())
len(fg.neighbors)
len(fg)
fg.edges
fg.edges(1)
len(fg.edges)
len(fg.edges())
fg.edge(1)
fg.edge[1]
fg.edge[2]
fg.edge
get_ipython().magic(u'clear ')
fg = nx.read_edgelist('./data/facebook_combined.txt')
fg.edge[1]
fg.edges[1]
fg.edges
fg.edges()
facebook_graph
fb_edgelist = str(facebook_graph.read()).split()
len(fb_edgelist)
facebook_graph.read()
facebook_graph.seek(0)
fb_edgelist = str(facebook_graph.read()).split()
fb_edgelist
facebook_graph.seek(0)
fb_edgelist = str(facebook_graph.read()).split('\n')
fb_edgelist[0]
fb_edgelist[1]
len(fb_edgelist)
facebook_graph = nx.Graph(fb_edgelist)
fb_edgelist = [tuple(l) for l in fb_edgelist]
fb_edgelist[0]
list((0,1,2))
get_ipython().magic(u'save 2016-06-29_ipython_log 1-112')
