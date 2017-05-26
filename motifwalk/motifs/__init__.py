import networkx as nx
try:
    import graph_tool as gt
except ImportError:
    print("Warning: graph_tool module is missing, motif analysis \
          is not available")

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
        self.anchors = anchors
        self.name = name

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

m4_0 = Motif([(1,0), (0,1), (2,0), (0,2), (3,0), (0,3), (1,2), (2,1),
              (1,3), (3,1), (2,3), (3,2)], is_directed=True, name="m4_0")
