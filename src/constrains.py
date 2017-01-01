import networkx as nx
import numpy as np
from numpy.random import choice


class Constrains(object):

    """Contrains object to guide the random walk
    based on the current context."""

    def __init__(self):
        pass

    def select(self, **kwargs):
        pass


class R(Constrains):

    """Random walk constrain."""

    def __init__(self):
        super().__init__()
        self._desc = """Random walk with no constrain."""

    def select(self, curr_node, graph, data):
        
        """Select next random node in the graph
           based on the current nodes."""

        return choice(list(graph[curr_node]))


class N2V(Constrains):

    """Node2Vec biased random walk."""

    def __init__(self, p=0.25, q=0.25):
    
        """Create a constrain object as described
        in the Node2Vec paper."""

        super().__init__()
        self._desc = """Node2Vec walk with p and q."""
        self._p = p 
        self._q = q
        self._alias_is_set = False
        
    def select(self, curr_node, graph, data):

        """Select next random node in the graph
           based on the defined hyperparameter
           p and q."""     

    def _alias_setup(self, probs):

        """Compute utility lists for non-uniform sampling from
        discrete distributions. Ref: hips.seas.harvard.edu"""

        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self._J = J
        self._q = q
        self._alias_is_set = True
