import pickle
import numpy as np
import utils
import networkx as nx
from motifwalk.utils.Graph import GraphContainer
from motifwalk.motifs import Motif
try:
    from graph_tool.clustering import motifs, motif_significance
    from graph_tool.spectral import adjacency
    from graph_tool import load_graph_from_csv
except ImportError:
    print("Warning: graph_tool module is missing, motif analysis \
          is not available")
