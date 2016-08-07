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
from matplotlib import colors

import pickle

import os
import time

from keras.optimizers import Adam
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

dataset_name = "egonets"
index_cols = True
epoch = 1
emb_dim = 10
neg_samp = 3
num_skip = 3
num_walk = 2
walk_length = 4
window_size = 3
iters = 1
num_batches = 100

is_label = True
rand_train = True
motif_train = True

fb = g.graph_from_pickle('data/{}.graph'.format(dataset_name))

try:
    correct_labels = util.read_correct_labels("data/raw/{}_labels.txt"
                                        .format(dataset_name), index_cols)
except:
    correct_labels = [0] * len(fb.nodes())



exp_name = "nce_{}_e{}_ed{}_ne{}_ns{}_nw{}_wl{}_ws{}_it{}_nb{}_adam".format(
    dataset_name, epoch, emb_dim, neg_samp, num_skip, num_walk,
    walk_length, window_size, iters, num_batches)


name_rand = exp_name + '_rand'
name_motif = exp_name + '_motif'


model_r = e.EmbeddingNet(graph=fb, epoch=epoch, emb_dim=emb_dim,
                            neg_samp=neg_samp, num_skip=num_skip,
                            num_walk=num_walk, walk_length=walk_length,
                            window_size=window_size, iters=iters)
model_r.build(optimizer='adam')
print("start training random walk")
start = time.time()
model_r.train(mode='random_walk', num_batches=num_batches,
                save_dir=os.path.join("weights", name_rand))
time_r = time.time() - start
print('finish training: Time: {}[s]'.format(time_r))
weight_r = model_r._model.get_weights()


model_m = e.EmbeddingNet(graph=fb, epoch=epoch, emb_dim=emb_dim,
                            neg_samp=neg_samp, num_skip=num_skip,
                            num_walk=num_walk, walk_length=walk_length,
                            window_size=window_size, iters=iters)
model_m.build(optimizer='adam')
print("start training motif walk")
start = time.time()
model_m.train(mode='motif_walk', num_batches=num_batches,
                save_dir=os.path.join("weights", name_motif))
time_m = time.time() - start
print("finish training: Time {}[s]".format(time_m))
weight_m = model_m._model.get_weights()


# Normalize
weight_r_avg = normalize(weight_r[0])
weight_m_avg = normalize(weight_m[0])

tsne_weight_r_norm = TSNE(learning_rate=300).fit_transform(weight_r_avg)
tsne_weight_m_norm = TSNE(learning_rate=300).fit_transform(weight_m_avg)

color_map = list(colors.cnames.keys())
colors = [color_map[c%len(color_map)] for c in correct_labels]

fig = plt.figure(figsize=(15, 45))
fig.suptitle(name_rand[:-5], fontsize=16)
a = plt.subplot(311)
a.set_title("Random walk embedding")
a.scatter(tsne_weight_r_norm[:, 0], tsne_weight_r_norm[:, 1],
          c=colors, s=30)
#for i, xy in enumerate(tsne_weight_r_norm):
#    a.annotate('%s' % (i + 1), xy=xy, textcoords='data')
b = plt.subplot(312)
b.set_title("Motif walk embedding")
b.scatter(tsne_weight_m_norm[:, 0], tsne_weight_m_norm[:, 1],
          c=colors, s=30)
#for i, xy in enumerate(tsne_weight_m_norm):
#    b.annotate('%s' % (i + 1), xy=xy, textcoords='data')
plt.savefig(name_rand[:-5] + '.png')
plt.show()
