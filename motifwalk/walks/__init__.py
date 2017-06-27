import numpy as np
import threading
from numpy.random import randint, shuffle, seed
from itertools import islice

seed(42)


def undirected_randomwalk(gt, walk_length=80, num_walk=10):
    """Perform random walk on graph gt and return the list of nodes.
    The input graph gt must be preprocessed to have contiguous node ids.
    WARNING: This function sets the provided graph to undirected if
    it is not initially undirected, then by the end of the random walk,
    the original directedness will be recovered.

    Parameters:
    gt - graph_tool.Graph - The undirected graph to walk on
    walk_length - int - Length of each walk
    num_walk - number of walks starting from each node

    Returns:
    context - np.ndarray - Contain list of nodes in a 2D array
    ci - int - Pointer to the next data location in context
    """
    context = np.ndarray(shape=(gt.num_vertices() * num_walk * walk_length),
                         dtype=np.uint32)
    print("Generating context of size {}".format(context.size))
    ci = 0
    directedness = gt.is_directed()
    if gt.is_directed():
        gt.set_directed(False)
    nlist = [i for i in gt.vertices()] # Only get valid nodes
    for walk in range(num_walk):
        shuffle(nlist)
        for v in nlist:
            context[ci] = int(v)
            ci += 1
            # walk_length-1 because we registered the first node
            for step in range(walk_length-1):
                next_index = randint(0, v.out_degree())
                v = next(islice(v.out_neighbours(), next_index, next_index+1))
                context[ci] = int(v)
                ci += 1
    gt.set_directed(directedness)
    return context, ci


class WalkThread(threading.Thread):
    """Thread class for parallel random walk context generation."""
    def __init__(self, threadID, start_index, end_index,
                 input_arr, output_arr, target_func):
        super().__init__()
        self.threadID = threadID
        self.si = start_index
        self.ei = end_index
        self.i = input_arr
        self.o = output_arr
        self.f = target_func
    def run(self):
        for i in range(self.start_index, self.end_index):
            self.o[i] = self.f(self.i[i])

def parallel_urw(gt, num_thread=8):
    """TODO: Implement parallel undirected random walk"""
    all_threads = []
    for i in range(num_thread):
        new_thread = WalkThread(i, i * elem_per_thread, (i+1) * elem_per_thread,
                                input_graph, data, undirected_randomwalk_kernel)
        all_threads.append(new_thread)
