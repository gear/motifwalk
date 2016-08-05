"""Generate pickle batch
"""
# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import graph as g
import multiprocessing as mp
import pandas as pd 

__author__ = "Shun Nukui"
__email__ = "nukui.s@ai.cs.titech.ac.jp"



dataset_name = "youtube"
mode = "random_walk"
num_batch = 1000

epoch = 1
neg_samp = 5
num_skip = 5
num_walk = 5
walk_length = 15
window_size = 5
iters = 50
num_batches = 1000
proc = 8
max_iter = 100

walk_basename = "walks/{}_e{}_ne{}_ns{}_nw{}_wl{}_ws{}_it{}_nb{}".format(
    dataset_name, epoch, neg_samp, num_skip, num_walk,
    walk_length, window_size, iters, num_batches)

if not os.path.exists(walk_basename):
    os.makedirs(walk_basename)

fb = g.graph_from_pickle('data/{}.graph'.format(dataset_name))



def generate_pickle_batch(i):
    gen_walk = fb.gen_walk(walk_func_name=mode, num_batches=num_batches,
                        walk_length=walk_length, num_walk=num_walk,
                        neg_samp=neg_samp, num_skip=num_skip, 
                        window_size=window_size)
    for n in range(max_iter):
        walk_name = os.path.join(walk_basename, "{}_{}.pkl".format(i, n))
        obj = next(gen_walk)
        pd.to_pickle(obj, walk_name)
        print("pickled "+ walk_name)


pool = mp.Pool(proc)

pool.map(generate_pickle_batch, range(1,proc+1))

