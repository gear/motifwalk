import pickle
from itertools import combinations
import numpy as np
import networkx as nx
from motifwalk.utils.Graph import GraphContainer
from motifwalk.motifs import all_u3, all_3, all_u4, all_4
from graph_tool.all import Graph, motifs, GraphView

def count_motif(g, motif_object, rm=True):
    """Count the number of given motif and return the mapping list
    to the vertices belongs to the motif. Note that `motifs` function
    return a triplet, lossing pointer to the first element in This
    triplet will make all PropertyMap orphan.

    Parameters:
    graph_container - GraphContainer - Graph storage object
    motif_object - Motif - Motif container
    rm - boolean - Return Maps

    Returns:
    motifs - list - contains gt.Graph objects
    counts - list - corresponding counts for motifs
    vertex_map - list - contains lists of gt.PropertyMap
    """
    m = motif_object.gt_motif
    # graph_tool.clustering.motifs
    rm, c, v_map = motifs(g, m.num_vertices(), motif_list=[m], return_maps=rm)
    return rm, c, v_map

def construct_motif_graph(graph_container, motif, vertex_maps=None):
    """Construct and return a undirected gt graph containing
    motif relationship. Note that graph_tool generates empty nodes
    to fill in the missing indices. For example, if we add edge (1,2)
    to an empty graph, the graph will have 3 nodes: 0, 1, 2 and 1 edge (1,2).
    For this reason, the returned `m_graph` usually has a large number of
    disconnected nodes.

    Parameters:
    graph_container - GraphContainer - Store the original network
    motif - Motif - Motif in study

    Returns:
    m_graph - gt.Graph - Undirected graph for motif cooccurence
    """
    if motif.anchors is None:
        print("Warning: Turning motif groups into cliques.")
    graph = graph_container.get_gt_graph()
    # graph_tool.Graph
    m_graph = Graph(directed=False)
    if vertex_maps is None:
        m, c, vertex_maps = count_motif(graph, motif)
    for prop_list in vertex_maps:
        for prop in prop_list:
            edges = [i for i in motif.anchored_edges(graph, prop.get_array())]
            m_graph.add_edge_list(edges)
    return m_graph

def filter_isolated(gt):
    """Filter isolated nodes (zero degrees) and return a GraphView. This
    function is for the purpose of shit

    Parameters:
    gt - graph_tool.Graph - network

    Returns:
    gt_filtered - graph_tool.GraphView - network containing only connected nodes
    """
    zero_degree_filter = gt.new_vertex_property("bool")
    for i in gt.vertices():
        v = gt.vertex(i)
        if v.out_degree() > 0 or v.in_degree() > 0:
            zero_degree_filter[i] = True
        else:
            zero_degree_filter[i] = False
    return GraphView(gt, zero_degree_filter)
