import networkx as nx
import numpy as np
from numpy.random import choice

class Constrains(object):

    '''Contrains object to guide the random walk
    based on the current context.'''

    def __init__(self):
        pass

    def select(self):
        pass

class R(Constrains):

    '''Random walk constrain.'''

    def __init__(self):
        self._desc = '''Random walk with no constrain.'''

    def select(self, curr_node, graph, data):
        
        '''Select next random node in the graph
           based on the current nodes.'''

        return choice(list(graph[curr_node]))

class N2V(Constrains):

    '''Node2Vec biased random walk.'''

    def __init__(self, p = 0.25, q = 0.25):
    
        '''Create a constrain object as described
        in the Node2Vec paper.'''
    
        self._desc = '''Node2Vec walk with p and q.'''
        self._p = p 
        self._q = q
        
    def select(self, curr_node, graph, data):

        '''Select next random node in the graph
           based on the defined hyperparameter
           p and q.'''     
    
        
