import networkx as nx
import numpy as np
from numpy.random import choice, rand
from abc import ABCMeta, abstractmethod


class Constrains(object):

    """Contrains object to guide the random walk
    based on the current context."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def select(self, **kwargs):
        pass


class R(Constrains):

    """Random walk constrain."""

    def __init__(self):
        super().__init__()
        self._desc = """Random walk with no constrain."""

    def select(self, curr_node, graph):
        
        """Select next random node in the graph
           based on the current nodes."""

        return choice(list(graph[curr_node]))

class UTriangle(Constrains):

    """Random walk in triangle manner."""

    def __init__(self, enforce_prob=0.9):

        """Create a triangle motif constrain object with the
        probability of following the triangle pattern is
        `enforce_prob`."""

        super().__init__()
        self._a = enforce_prob

    def select(self, curr_node, graph, data):

        """Select the next node in based on current
        node and the current walk."""

        triangle_nodes = [i for i in graph[curr_node]
                       if set().union(graph[i], graph[curr_node])]
        rand_num = rand()
        if len(triangle_nodes) > 0:
            if rand_num > self._a:
                return choice(triangle_nodes)
        return choice(graph.neighbors(curr_node))


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

    def select(self, curr_node, nxgraph, data):

        """Select next random node in the graph
        based on the defined hyperparameter
        p and q."""


    def _preprocess_transition_probs(self, nxgraph):

        """Preprocessing of transition probs
        for guiding the random walks."""

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [nxgraph[node][nbr]['weight'] for
                                  nbr in sorted(nxgraph.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = _alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if nxgraph.is_directed():
            for edge in nxgraph.edges():
                alias_edges[edge] = self._get_alias_edge(graph, edge[0], edge[1])
        else:
            pass

def test():
    print(nx)

def _alias_setup(self, probs):

    """Compute utility lists for non-uniform sampling from
    discrete distributions. Ref: hips.seas.harvard.edu"""

    k = len(probs)
    q = np.zeros(k)
    j = np.zeros(k, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = k*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return j, q

def _alias_draw(j, q):

    """Draw sample from a non-uniform discrete distribution
	using alias sampling."""

    k = len(j)

    kk = int(np.floor(np.random.rand()*k))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]

if __name__ == '__main__':
    test()