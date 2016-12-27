import networkx as nx
import utils
import numpy as np

class WalkGenerator(object):
    """
    Generic graph path generator.
    """
    def __init__(self, graph = None, constrain = None):
        """
        Create WalkGenerator object with associcated
        graph and the constrain for the walk.
        """
        self.g = graph
        self.constrain = constrain 
