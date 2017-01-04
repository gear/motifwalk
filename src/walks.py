import networkx as nx
import numpy as np
from numpy.random import choice
from numpy.random import randint
from constrains import N2V, R, UTriangle, UWedge
# from mage.utils import *
# from mage.constrains import N2V, R # Node2Vec walk, Random Walk


class WalkGenerator(object):

    """Generic graph path generator. """

    def __init__(self, graph=None, constrain=None):

        """Create WalkGenerator object with associcated
        graph and the constrain object for the walk.
        
        Example:
            G = nx.graph('amazon_edgelist')
            n2v = Progressive(p=0.25, q=0.25)
            corpus_gen = WalkGenerator(G, n2v)"""

        self.g = graph
        self.c = constrain 

    def setup(self, graph=None, constrain=None):
    
        """ Set the graph or constrain to new values. """

        self.g = graph
        self.c = constrain

    def __call__(self, walk_length=10, num_walk=10,
                 batch=10, start=None, reset=None):
        """ Short cut for calling the walk generator. """
        
        return self._gen(walk_length, num_walk,
                         batch, start, reset)

    def _gen(self, walk_length=10, num_walk=10,
             start=None, reset=None):
        
        """ Generate batches of random walk node ids.

        Parameters:
            walk_length: (int, tuple, list) specify how the
                walk's length is computed.
            num_walk: (int) number of walks in total
            start: (int, str) graph (networkx) node id.

        Example:
            walk_generator._gen() => Generate 1 batch 
                                     of 10 random walks 
                                     of length 10. """

        # Determine start node
        if start is None or start not in self.g.nodes():
            start = choice(self.g)
 
        # Random walk with constrain and append the result  
        data = [start]
        next_node = start
        while num_walk > 0:
            for w in range(walk_length-1):
                # Using a constrains object to select next node
                next_node = self.c.select(next_node, self.g, data)
                if not next_node:
                    next_node = choice(self.g)
                data.append(next_node)
            yield data
            if reset:
                data.clear()
                data.append(start)
            else:
                data.clear()
                data.append(next_node)  # Last appended to data
            num_walk -= 1


def test():
    test_graph = nx.Graph()  # Undirected
    karate = [[2,1],[3,1],[3,2],[4,1],[4,2],[4,3],[5,1],
              [6,1],[7,1],[7,5],[7,6],[8,1],[8,2],[8,3],[8,4],
              [9,1],[9,3],[10,3],[11,1],[11,5],[11,6],[12,1],
              [13,1],[13,4],[14,1],[14,2],[14,3],[14,4],[17,6],
              [17,7],[18,1],[18,2],[20,1],[20,2],[22,1],[22,2],
              [26,24],[26,25],[28,3],[28,24],[28,25],[29,3],[30,24],
              [30,27],[31,2],[31,9],[32,1],[32,25],[32,26],[32,29],
              [33,3],[33,9],[33,15],[33,16],[33,19],[33,21],[33,23],
              [33,24],[33,30],[33,31],[33,32],[34,9],[34,10],[34,14],
              [34,15],[34,16],[34,19],[34,20],[34,21],[34,23],[34,24],
              [34,27],[34,28],[34,29],[34,30],[34,31],[34,32],[34,33]]
    test_graph.add_edges_from(karate)
    walker = WalkGenerator(graph=test_graph, constrain=R())
    print([i[:] for i in walker._gen()])
    walker_triangle = WalkGenerator(graph=test_graph, constrain=UTriangle())
    print([i[:] for i in walker_triangle._gen()])
    walker_wedge = WalkGenerator(graph=test_graph, constrain=UWedge())
    print([i[:] for i in walker_wedge._gen()])

if __name__ == '__main__':
    test()