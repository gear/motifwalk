
from libcpp.vector cimport vector
from libcpp.map cimport map
import random

cdef class Data:
    """
    Data class (struct) to return vector sets for python
    """
    cdef public vector[int] targets
    cdef public vector[int] classes
    cdef public vector[int] labels
    cdef public vector[double] weights

    def __cinit__(self, targets, classes, labels, weights):
        self.targets = targets
        self.classes = classes
        self.labels = labels
        self.weights = weights


cdef int contains_in_vector(vector[int] vec, int x):
    """
    Check if x in vec
    """
    cdef int i
    for i in vec:
        if i == x:
            return 1
    return 0

cdef random_walk(map[int, vector[int]] neighbors, int length,
                  rand_seed=None, start_node=None,
                  double reset=0.0):
    random.seed(rand_seed)
    cdef num_nodes = neighbors.size()
    cdef int start
    if start_node is None:
        start = random.randint(0, num_nodes-1)
    cdef vector[int] walk_path = [start]
    cdef int cur
    cdef int rnd
    cdef vector[int] cur_neighbor
    while walk_path.size() < length:
        cur = walk_path[-1]
        if neighbors[cur].size() > 0:
            if random.random() >= reset:
                rnd = random.randint(0, neighbors[cur].size() - 1)
                walk_path.push_back(neighbors[cur][rnd])
            else:
                walk_path.push_back(walk_path[0])
        else:
            break
    return walk_path


cdef vector[int] motif_walk(map[int, vector[int]] neighbors, int length,
                  int rand_seed=-1, int start_node=-1,
                  double reset=0.0, double walk_bias=0.9):
    if rand_seed > 0:
        random.seed(rand_seed)
    cdef num_nodes = neighbors.size()
    cdef int start
    if start_node < 0:
        start = random.randint(0, num_nodes-1)
    cdef vector[int] walk_path = [start]
    cdef int cur
    cdef int rnd
    cdef vector[int] cur_neighbor
    cdef int prev = -1
    cdef int cand
    cdef double prob
    cur = start
    while walk_path.size() < length:
        cand_ind = random.randint(0, neighbors[cur].size() - 1)
        cand = neighbors[cur][cand_ind]
        if prev:
            while True:
                prob = random.random()
                if contains_in_vector(neighbors[prev], cand):
                    if prob < walk_bias:
                        walk_path.push_back(cand)
                        break
                else:
                    if prob > walk_bias:
                        walk_path.push_back(cand)
                        break
                cand_ind = random.randint(0, neighbors[cur].size() - 1)
                cand = neighbors[cur][cand_ind]
        else:
            walk_path.push_back(cand)
        prev = cur
        cur = cand
    return walk_path

cpdef gen_walk_fast(map[int, vector[int]] neighbors, vector[int] freq_list,
                    walk_func_name, int num_batches=100, int walk_length=10,
                    int num_walk=5, int num_true=1, int neg_sample=15,
                    int num_skip=2, int shuffle=1, window_size=3,
                    double gamma=0.8, int rand_seed=-1):
    cdef int num_nodes = neighbors.size()
    cdef int num_freq = freq_list.size()
    cdef int cnt = 0
    cdef int j, lower, upper
    cdef int target, cls_node
    cdef int distance
    cdef vector[int] targets, classes
    cdef vector[int] labels
    cdef vector[double] weights
    cdef vector[int] walk
    while True:
        start = random.randint(0, num_nodes - 1)
        if not neighbors[start].size() > 0:
            continue

        if walk_func_name == "random_walk":
            walk = random_walk(neighbors, length=walk_length,
                               rand_seed=rand_seed, start_node=start)
        elif walk_func_name == "motif_walk":
            walk = motif_walk(neighbors, length=walk_length,
                              rand_seed=rand_seed, start_node=start)
        else:
            raise ValueError("walk_func_name is invalid")

        for j in range(walk.size()):
            target = walk[j]
            lower = max(0, j - window_size)
            upper = min(walk_length, j + window_size + 1)
            for _ in range(num_skip):
                cls_ind = random.randint(lower, upper - 1)
                distance = abs(j - cls_ind)
                cls_node = walk[cls_ind]
                targets.push_back(target)
                classes.push_back(cls_node)
                labels.push_back(1)
                weights.push_back(pow(gamma, distance))
            for _ in range(neg_sample):
                cls_node = freq_list[random.randint(0, num_freq - 1)]
                targets.push_back(target)
                classes.push_back(cls_node)
                labels.push_back(0)
                weights.push_back(1.0)
        if cnt >= num_batches:
            return Data(targets, classes, labels, weights)
        cnt += 1

