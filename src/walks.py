import networkx as nx
import utils
import numpy as np
from numpy.random import choice
from numpy.random import randint

class WalkGenerator(object):

    '''Generic graph path generator.  '''

    def __init__(self, graph = None, constrain = None):

        ''' Create WalkGenerator object with associcated
        graph and the constrain object for the walk.
        
        Example:
            G = nx.graph('amazon_edgelist')
            n2v = Progressive(p=0.25, q=0.25)
            corpus_gen = WalkGenerator(G, n2v) '''

        self.g = graph
        self.constrain = constrain 

    def setup(self, graph = None, constrain = None):
    
        ''' Set the graph or constrain to new values. '''

        self.g = graph
        self.constrain = constrain

    def __call__(self):
    
        ''' Short cut for calling the walk generator. '''
        
        return self._gen

    def _gen(self, walk_length = 10, num_walk = 10, \
                 batch = 10, start = None, reset = None):
        
        ''' Generate batches of random walk node ids.

        Parameters:
            walk_length: (int, tuple, list) specify how the
                walk's length is computed.
            num_walk: (int) number of walks in total 
            batch: (int) number of walks in each batch.
            start: (int, str) graph (networkx) node id.

        Example:
            walk_generator._gen() => Generate 1 batch 
                                     of 10 random walks 
                                     of length 10. '''

        # Function to determine walk length for each walk
        lengthf = lambda x: x
        # If walk_length is a tuple, choose a random integer
        # in the range that the tuple describes. If the walk
        # is a list, each walk with have the length specified
        # in the list.
        if isinstance(walk_length, tuple):
            lengthf = lambda x: randint(x[0], x[1])
        elif isinstance(walk_length, list):
            assert len(walk_length) == num_walk, \ 
                   'Number of walk and walk length list does not match'
            walk_length = (i for i in walk_length)
            lengthf = lambda x: next(x)
        else:
            pass

        # Determine start node
        if start is None or start not in self.graph.nodes():
            start = choice(self.graph.nodes())
 
        # Random walk with constrain and append the result  
        data = []
        for _ in num_walk: 
            while len(data) < batch:
                constrain.select(curr_node)
