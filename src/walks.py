import networkx as nx
import numpy as np
from numpy.random import choice, shuffle
from constrains import R, UTriangle, UWedge
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

    def __call__(self, walk_length=10, num_walk=10, yield_size=None):
        """ Short cut for calling the walk generator. """
        
        return self._gen(walk_length, num_walk, yield_size)

    def _gen(self, walk_length=10, num_walk=1, yield_size=None):
        
        """ Generate batches of random walk node ids.

        Parameters:
            walk_length: Length of a walk starting with a node.
            num_walk: (int) number of walks for each node.
            yield_size: Data buffer size

        Example:
            walk_generator._gen() => Generate 1 batch 
                                     of 10 random walks 
                                     of length 10. """

        total_length = len(self.g) * num_walk * walk_length
        context_length = yield_size if yield_size else total_length
        if context_length > total_length:
            context_length = total_length
        buffer = np.ndarray(shape=context_length, dtype=np.int32)
        i = 0
        print("Walking total of {} walks, will yield every {} nodes...".format(
            len(self.g) * num_walk, context_length
        ))
        for _ in range(num_walk):
            nodes = self.g.nodes()[:]
            shuffle(nodes)
            for node in nodes:
                buffer[i] = node
                i += 1
                if i % context_length == 0:
                    yield buffer
                    i = 0
                next_node = node
                for w in range(walk_length-1):
                    next_node = self.c.select(next_node, self.g, buffer, i)
                    if not next_node:
                        next_node = choice(self.g)
                    buffer[i] = next_node
                    i += 1
                    if i % context_length == 0:
                        yield buffer
                        i = 0
        if i != 0:  # Yield the remaining data
            buffer[i:].fill(-1)
            yield buffer

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
    print([i for i in walker._gen()])
    walker_triangle = WalkGenerator(graph=test_graph, constrain=UTriangle())
    print(i for i in walker_triangle._gen())
    walker_wedge = WalkGenerator(graph=test_graph, constrain=UWedge())
    print(i for i in walker_wedge._gen())

if __name__ == '__main__':
    test()