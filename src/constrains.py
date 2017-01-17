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

    def select(self, curr_node, graph, data=None, di=None):
        
        """Select next random node in the graph
           based on the current nodes."""

        return choice(graph.neighbors(curr_node))


class UTriangle(Constrains):

    """Random walk in triangle manner."""

    def __init__(self, enforce_prob=0.9):

        """Create a triangle motif constrain object with the
        probability of following the triangle pattern is
        `enforce_prob`."""

        super().__init__()
        self._a = enforce_prob
        self._num_nodes = 3
        self._is_directed = False

    def select(self, curr_node, graph, data=None, di=None):

        """Select the next node in based on current
        node and the current walk."""

        if graph.is_directed() != self._is_directed:
            print("Warning: Using mismatch directness.")

        triangle_nodes = [i for i in graph[curr_node]
                          if set().union(graph[i], graph[curr_node])]
        rand_num = rand()
        if len(triangle_nodes) > 0:
            if rand_num > self._a:
                return choice(triangle_nodes)
        return choice(graph.neighbors(curr_node))


class UWedge(Constrains):

    """Random walk in wedge manner."""

    def __init__(self, enforce_prob=0.9):

        """Create a wedge motif constrain object with the
        probability of following the triangle pattern is
        `enforce_prob`."""

        super().__init__()
        self._a = enforce_prob
        self._num_nodes = 3
        self._is_directed = False

    def select(self, curr_node, graph, data=None, di=None):

        """Select the next node based on current
        node and the current walk."""

        if graph.is_directed() != self._is_directed:
            print("Warning: Using mismatch directness.")

        if not graph[curr_node]:
            return choice(graph)

        wedge_nodes = [i for i in graph[curr_node]
                       if not set().union(graph[i], graph[curr_node])]
        rand_num = rand()
        if len(wedge_nodes) > 0:
            if rand_num > self._a:
                return choice(wedge_nodes)
        return choice(graph.neighbors(curr_node))

def test():
    print("Constrain module.")

if __name__ == '__main__':
    test()
