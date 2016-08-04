"""Warpper for testing
"""
# Coding: utf-8
# File name: test.py
# Created: 2016-07-27
# Description:
# v0.0: File created.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import embeddings as e
import graph as g
import util

import numpy as np
import cPickle as p
from matplotlib import colors

import os
import time

from keras.optimizers import Adam
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


dataset_name = "polblogs"
index_col = False
epoch = 1
emb_dim = 50
neg_samp = 10
num_skip = 2
num_walk = 5
walk_length = 5
window_size = 5
iters = 50
num_batches = 10000

is_label = True
rand_train = True
motif_train = True


correct_labels = util.read_correct_labels("data/raw/{}_labels.txt"
                                        .format(dataset_name),
                                        index_col)


fb = g.graph_from_pickle('data/{}.graph'.format(dataset_name))

exp_name = "nce_{}_e{}_ed{}_ne{}_ns{}_nw{}_wl{}_ws{}_it{}_nb{}_adam".format(
    dataset_name, epoch, emb_dim, neg_samp, num_skip, num_walk,
    walk_length, window_size, iters, num_batches)


name_contrast = exp_name + '_contrast'


model_c = e.EmbeddingNet(graph=fb, epoch=epoch, emb_dim=emb_dim,
                            neg_samp=neg_samp, num_skip=num_skip,
                            num_walk=num_walk, walk_length=walk_length,
                            window_size=window_size, iters=iters)
model_c.build(optimizer='adam')
print("start training contrast walk")
start = time.time()
model_c.train_mce(pos='motif_walk', neg="random_walk", num_batches=num_batches)
time_c = time.time() - start
print("finish training: Time {}[s]".format(time_c))
weight_c = model_c._model.get_weights()

if not os.path.exists(name_contrast + '.weights'):
    with open(name_contrast + '.weights', 'wb') as f:
        p.dump(weight_c, f, p.HIGHEST_PROTOCOL)

# Normalize
weight_c_avg = normalize(weight_c[0])

color_map = colors.cnames.keys()
colors = [color_map[c%len(color_map)] for c in correct_labels]


tsne_weight_c_norm = TSNE().fit_transform(weight_c_avg)

fig = plt.figure(figsize=(15, 45))
fig.suptitle(exp_name, fontsize=16)
a = plt.subplot(311)
a.set_title("contrast walk embedding")
a.scatter(tsne_weight_c_norm[:, 0], tsne_weight_c_norm[:, 1],
          c=colors, s=20)
plt.savefig(exp_name + '.png')
plt.show()

