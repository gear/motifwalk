import networkx as nx
import graph_tool as gt
from graph_tool.all import *
from itertools import combinations

class Motif:

    def __init__(self, edge_list, is_directed=False, anchors=None, name=None):
        nx_graph = nx.Graph()
        if is_directed:
            nx_graph = nx_graph.to_directed()
        nx_graph.add_edges_from(edge_list)
        gt_graph = gt.Graph()
        gt_graph.set_directed(is_directed)
        gt_graph.add_edge_list(edge_list)
        self.gt_motif = gt_graph
        self.nx_motif = nx_graph
        self.size = nx_graph.size()
        self.anchors = anchors
        self.name = name

    def anchored_edges(self, gt_graph, node_list):
        """Return a list of edges for motif graph construction

        Parameters:
        gt_graph - gt.Graph - the original graph
        node_list - list - isomorphism nodes to create induced subgraphs

        Returns:
        edge_list - list - edges for creating the motif graph
        """
        if self.anchors is None:
            return combinations(node_list, 2)
        if len(self.anchors) > self.size:
            raise ValueError("Anchors set is of invalid size")
        # Create a node filter
        vfilt = g.new_vertex_property('bool');
        for i in node_list:
            vfilt[i] = True
        # Subgraph as gt.GraphView
        induced_subgraph = GraphView(gt_graph, vfilt)
        # Get the motif mapping
        _, mapping = isomorphism(sub, self.gt_motif, isomap=True)
        # List of anchored nodes
        anchored = [i for i in node_list if mapping[i] in self.anchors]
        assert len(anchored) > 1, "There is not enough anchor nodes to\
                                   make an edge."
        return combinations(anchored, 2)

# List of connected motifs: m[size][directed] and their aliases

# Size 3 motifs

m3u_0 = Motif([(0,1), (1,2), (2,0)], name="m3u_0")
triangle = m3u_0

m3u_1 = Motif([(0,1), (0,2)], name="m3u_1")
wedge = m3u_1

m3_0 = Motif([(0,1), (1,0), (1,2), (2,1), (0,2), (2,0)],
             is_directed=True, name="m3_0")
m3_1 = Motif([(0,1), (1,0), (2,1), (0,2), (2,0)],
             is_directed=True, name="m3_1")
m3_2 = Motif([(0,1), (1,0), (2,0), (0,2)],
             is_directed=True, name="m3_2")
m3_3 = Motif([(0,1), (1,0), (0,2), (1,2)],
             is_directed=True, name="m3_3")
m3_4 = Motif([(0,1), (1,0), (2,0), (1,2)],
             is_directed=True, name="m3_4")
m3_5 = Motif([(0,1), (1,0), (2,0), (2,1)],
             is_directed=True, name="m3_5")
m3_6 = Motif([(0,1), (1,0), (2,0)],
             is_directed=True, name="m3_6")
m3_7 = Motif([(0,1), (1,0), (0,2)],
             is_directed=True, name="m3_7")
m3_8 = Motif([(0,1), (1,2), (2,0)],
             is_directed=True, name="m3_8")
m3_9 = Motif([(0,1), (2,1), (2,0)],
              is_directed=True, name="m3_9")
feed_forward = m3_9
m3_10 = Motif([(0,1), (2,0)],
              is_directed=True, name="m3_10")
m3_11 = Motif([(0,1), (0,2)],
              is_directed=True, name="m3_11")
m3_12 = Motif([(1,0), (2,0)],
              is_directed=True, name="m3_12")

# Size 4 motifs

m4u_0 = Motif([(0,1), (0,2), (0,3)], name="m4u_0")

m4u_1 = Motif([(0,1), (0,3), (1,2)], name="m4u_1")

m4u_2 = Motif([(0,1), (0,3), (3,2), (1,3)], name="m4u_2")

m4u_3 = Motif([(0,1), (0,3), (1,2), (3,2)], name="m4u_3")

m4u_4 = Motif([(0,1), (0,3), (1,3), (3,2), (1,2)], name="m4u_4")

m4u_5 = Motif([(0,1), (0,3), (1,3), (3,2), (1,2), (0,2)], name="m4u_5")

m4_0 = Motif([(1, 3), (1, 2), (1, 0)], is_directed=True, name='m4_0')
m4_1 = Motif([(0, 1), (0, 2), (3, 1)], is_directed=True, name='m4_1')
m4_2 = Motif([(0, 3), (2, 1), (2, 0)], is_directed=True, name='m4_2')
m4_3 = Motif([(2, 3), (3, 1), (3, 0)], is_directed=True, name='m4_3')
m4_4 = Motif([(0, 3), (1, 3), (2, 3)], is_directed=True, name='m4_4')
m4_5 = Motif([(0, 3), (1, 0), (2, 0)], is_directed=True, name='m4_5')
m4_6 = Motif([(1, 0), (2, 3), (3, 0)], is_directed=True, name='m4_6')
m4_7 = Motif([(0, 3), (1, 2), (2, 0)], is_directed=True, name='m4_7')
m4_8 = Motif([(1, 0), (1, 2), (1, 3), (2, 3)], is_directed=True, name='m4_8')
m4_9 = Motif([(2, 3), (3, 0), (3, 2), (3, 1)], is_directed=True, name='m4_9')
m4_10 = Motif([(0, 3), (0, 2), (1, 3), (1, 2)], is_directed=True, name='m4_10')
m4_11 = Motif([(0, 1), (0, 3), (3, 2), (3, 1)], is_directed=True, name='m4_11')
m4_12 = Motif([(1, 2), (1, 3), (3, 0), (3, 1)], is_directed=True, name='m4_12')
m4_13 = Motif([(0, 1), (0, 3), (2, 1), (3, 1)], is_directed=True, name='m4_13')
m4_14 = Motif([(0, 2), (1, 3), (1, 2), (2, 3)], is_directed=True, name='m4_14')
m4_15 = Motif([(0, 1), (1, 3), (2, 3), (2, 0)], is_directed=True, name='m4_15')
m4_16 = Motif([(0, 2), (2, 1), (2, 3), (3, 1)], is_directed=True, name='m4_16')
m4_17 = Motif([(0, 3), (1, 3), (2, 0), (2, 1)], is_directed=True, name='m4_17')
m4_18 = Motif([(0, 3), (1, 2), (1, 3), (2, 1)], is_directed=True, name='m4_18')
m4_19 = Motif([(1, 2), (2, 1), (3, 2), (3, 0)], is_directed=True, name='m4_19')
m4_20 = Motif([(1, 3), (1, 2), (2, 0), (3, 2)], is_directed=True, name='m4_20')
m4_21 = Motif([(0, 1), (1, 2), (1, 3), (3, 1)], is_directed=True, name='m4_21')
m4_22 = Motif([(0, 1), (0, 2), (2, 0), (3, 2)], is_directed=True, name='m4_22')
m4_23 = Motif([(0, 3), (0, 1), (1, 2), (2, 0)], is_directed=True, name='m4_23')
m4_24 = Motif([(1, 3), (2, 0), (3, 2), (3, 1)], is_directed=True, name='m4_24')
m4_25 = Motif([(0, 1), (1, 2), (2, 1), (3, 1)], is_directed=True, name='m4_25')
m4_26 = Motif([(0, 1), (1, 2), (2, 1), (3, 2)], is_directed=True, name='m4_26')
m4_27 = Motif([(0, 1), (1, 2), (2, 0), (3, 0)], is_directed=True, name='m4_27')
m4_28 = Motif([(0, 3), (1, 2), (2, 1), (3, 1)], is_directed=True, name='m4_28')
m4_29 = Motif([(0, 3), (1, 2), (2, 0), (3, 1)], is_directed=True, name='m4_29')
m4_30 = Motif([(2, 1), (2, 0), (3, 2), (3, 0), (3, 1)],
              is_directed=True, name='m4_30')
m4_31 = Motif([(1, 3), (1, 0), (1, 2), (2, 3), (3, 0)],
              is_directed=True, name='m4_31')
m4_32 = Motif([(0, 1), (0, 2), (2, 3), (2, 1), (3, 1)],
              is_directed=True, name='m4_32')
m4_33 = Motif([(0, 3), (2, 1), (2, 3), (3, 0), (3, 1)],
              is_directed=True, name='m4_33')
m4_34 = Motif([(1, 0), (2, 3), (2, 1), (3, 1), (3, 0)],
              is_directed=True, name='m4_34')
m4_35 = Motif([(0, 3), (1, 0), (1, 2), (3, 2), (3, 1)],
              is_directed=True, name='m4_35')
m4_36 = Motif([(0, 1), (2, 3), (2, 1), (3, 2), (3, 0)],
              is_directed=True, name='m4_36')
m4_37 = Motif([(0, 2), (1, 2), (1, 0), (2, 3), (3, 2)],
              is_directed=True, name='m4_37')
m4_38 = Motif([(0, 2), (1, 0), (2, 1), (3, 1), (3, 2)],
              is_directed=True, name='m4_38')
m4_39 = Motif([(0, 3), (1, 2), (2, 0), (2, 1), (3, 1)],
              is_directed=True, name='m4_39')
m4_40 = Motif([(0, 2), (1, 3), (2, 1), (2, 0), (3, 1)],
              is_directed=True, name='m4_40')
m4_41 = Motif([(0, 2), (0, 1), (1, 3), (2, 3), (3, 0)],
              is_directed=True, name='m4_41')
m4_42 = Motif([(0, 3), (1, 0), (1, 2), (2, 0), (3, 1)],
              is_directed=True, name='m4_42')
m4_43 = Motif([(0, 2), (1, 2), (2, 3), (2, 0), (3, 1)],
              is_directed=True, name='m4_43')
