from libcpp.vector cimport vector
from libcpp.map cimport map

import random

cdef class Data:
    """
    Data class (struct) to return vector sets for python
    """
    cdef public vector[long] targets
    cdef public vector[long] classes
    cdef public vector[int] labels
    cdef public vector[double] weights

    def __cinit__(self, targets, classes, labels, weights):
        self.targets = targets
        self.classes = classes
        self.labels = labels
        self.weights = weights


cdef int contains_in_vector(vector[long] vec, long x):
    """
    Check if x in vec
    """
    cdef long i
    for i in vec:
        if i == x:
            return 1
    return 0

cdef vector[long] random_walk(map[long, vector[long]] neighbors, long length,
                            long rand_seed=-1, long start_node=-1,
                            double reset=0.0):
    if rand_seed > 0:
        random.seed(rand_seed)
    cdef num_nodes = neighbors.size()
    cdef long start
    if start_node < 0:
        start = random.randint(0, num_nodes-1)
    else:
        start = start_node
    cdef vector[long] walk_path = [start]
    cdef long cur
    cdef long rnd
    cur = start
    while walk_path.size() < length:
        if neighbors[cur].size() > 0:
            if random.random() >= reset:
                rnd = random.randint(0, neighbors[cur].size() - 1)
                walk_path.push_back(neighbors[cur][rnd])
            else:
                walk_path.push_back(walk_path[0])
        else:
            break
    return walk_path


cdef vector[long] motif_walk(map[long, vector[long]] neighbors, long length,
                  long rand_seed=-1, long start_node=-1,
                  double reset=0.0, double walk_bias=0.9):
    if rand_seed > 0:
        random.seed(rand_seed)
    cdef num_nodes = neighbors.size()
    cdef long start
    if start_node < 0:
        start = random.randint(0, num_nodes-1)
    else:
        start = start_node
    cdef vector[long] walk_path = [start]
    cdef long cur
    cdef long rnd
    cdef long prev = -1
    cdef long cand
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

cpdef Data gen_walk_fast(map[long, vector[long]] neighbors, vector[long] freq_list,
                        walk_func_name, long num_batches=100, long walk_length=10,
                        long num_walk=5, long num_true=1, long neg_sample=15,
                        long num_skip=2, long shuffle=1, window_size=3,
                        double gamma=0.8, long rand_seed=-1):
    cdef long num_nodes = neighbors.size()
    cdef long num_freq = freq_list.size()
    cdef long cnt = 0
    cdef long j, lower, upper
    cdef long target, cls_node
    cdef long distance
    cdef vector[long] targets, classes
    cdef vector[int] labels
    cdef vector[double] weights
    cdef vector[long] walk
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

