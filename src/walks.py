import networkx as nx
import numpy as np
from numpy.random import choice
from numpy.random import randint
from constrains import N2V, R
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
             batch=10, start=None, reset=None):
        
        """ Generate batches of random walk node ids.

        Parameters:
            walk_length: (int, tuple, list) specify how the
                walk's length is computed.
            num_walk: (int) number of walks in total 
            batch: (int) number of walks in each batch.
            start: (int, str) graph (networkx) node id.

        Example:
            walk_generator._gen() => Generate 1 batch 
                                     of 10 random walks 
                                     of length 10. """

        # Determine start node
        if start is None or start not in self.g.nodes():
            start = choice(self.g.nodes())
 
        # Random walk with constrain and append the result  
        data = [start]
        next_node = start
        while num_walk > 0:
            for b in range(batch):
                for w in range(walk_length):
                    # Using a constrains object to select next node
                    next_node = self.c.select(self.g, data)
                data.append(next_node)
            yield data
            if reset:
                data.clear()
                data.append(start)
            else:
                data.clear()
                data.append(next_node)  # Last appended to data
            num_walk -= 1