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

import numpy as np
import cPickle as p

import os
import time

from keras.optimizers import Adam
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


rand_train = True
motif_train = True
contrast_train = True

save_rand = True
save_motif = True
save_contrast = True

dataset_name = "polblogs"
epoch = 1
emb_dim = 10
neg_samp = 5
num_skip = 1
num_walk = 5
walk_length = 5
window_size = 2
iters = 10
num_batches = 10000




fb = g.graph_from_pickle('data/{}.graph'.format(dataset_name))

exp_name = "nce_{}_e{}_ed{}_ne{}_ns{}_nw{}_wl{}_ws{}_it{}_nb{}_adam".format(
    dataset_name, epoch, emb_dim, neg_samp, num_skip, num_walk,
    walk_length, window_size, iters, num_batches)


name_rand = exp_name + '_rand'
name_motif = exp_name + '_motif'
name_contrast = exp_name + '_contrast'


if not rand_train:
    weight_r = p.load(open(name_rand+".weights", "rb"))
else:
    model_r = e.EmbeddingNet(graph=fb, epoch=epoch, emb_dim=emb_dim,
                             neg_samp=neg_samp, num_skip=num_skip,
                             num_walk=num_walk, walk_length=walk_length,
                             window_size=window_size, iters=iters)
    model_r.build(optimizer='adam')
    print("start training random walk")
    start = time.time()
    model_r.train(mode='random_walk', num_batches=num_batches)
    time_r = time.time() - start
    print('finish training: Time: {}[s]'.format(time_r))
    weight_r = model_r._model.get_weights()

    # Save or load data
    if not os.path.exists(name_rand + '.weights') and rand_train:
        with open(name_rand + '.weights', 'wb') as f:
            p.dump(weight_r, f, p.HIGHEST_PROTOCOL)

if not motif_train:
    weight_m = p.load(open(name_motif+".weights", "rb"))
else:
    model_m = e.EmbeddingNet(graph=fb, epoch=epoch, emb_dim=emb_dim,
                             neg_samp=neg_samp, num_skip=num_skip,
                             num_walk=num_walk, walk_length=walk_length,
                             window_size=window_size, iters=iters)
    model_m.build(optimizer='adam')
    print("start training motif walk")
    start = time.time()
    model_m.train(mode='motif_walk', num_batches=num_batches)
    time_m = time.time() - start
    print("finish training: Time {}[s]".format(time_m))
    weight_m = model_m._model.get_weights()

    if not os.path.exists(name_motif + '.weights') and motif_train:
        with open(name_motif + '.weights', 'wb') as f:
            p.dump(weight_m, f, p.HIGHEST_PROTOCOL)

if not contrast_train:
    weight_c = p.load(open(name_contrast+".weights", "rb"))
else:
    model_c = e.EmbeddingNet(graph=fb, epoch=epoch, emb_dim=emb_dim,
                             neg_samp=neg_samp, num_skip=num_skip,
                             num_walk=num_walk, walk_length=walk_length,
                             window_size=window_size, iters=iters)  # reset default at 0.3
    model_c.build(optimizer='adam')
    print("start training constact walk")
    start = time.time()
    model_c.train_mce(num_batches=num_batches)
    time_c = time.time() - start
    print("finish training: Time {}[s]".format(time_c))
    weight_c = model_c._model.get_weights()

    if not os.path.exists(name_contrast + '.weights') and contrast_train:
        with open(name_contrast + '.weights', 'wb') as f:
            p.dump(weight_c, f, p.HIGHEST_PROTOCOL)

# Normalize
weight_r_avg = normalize(weight_r[0])
weight_m_avg = normalize(weight_m[0])
weight_c_avg = normalize(weight_c[0])

tsne_weight_r_norm = TSNE(learning_rate=100).fit_transform(weight_r_avg)
tsne_weight_m_norm = TSNE(learning_rate=100).fit_transform(weight_m_avg)
tsne_weight_c_norm = TSNE(learning_rate=100).fit_transform(weight_c_avg)

fig = plt.figure(figsize=(15, 45))
fig.suptitle(name_rand[:-5], fontsize=16)
a = plt.subplot(311)
a.set_title("Random walk embedding")
a.scatter(tsne_weight_r_norm[:, 0], tsne_weight_r_norm[:, 1])
#for i, xy in enumerate(tsne_weight_r_norm):
#    a.annotate('%s' % (i + 1), xy=xy, textcoords='data')
b = plt.subplot(312)
b.set_title("Motif walk embedding")
b.scatter(tsne_weight_m_norm[:, 0], tsne_weight_m_norm[:, 1])
#for i, xy in enumerate(tsne_weight_m_norm):
#    b.annotate('%s' % (i + 1), xy=xy, textcoords='data')
c = plt.subplot(313)
c.set_title("Contrast walk embedding")
c.scatter(tsne_weight_c_norm[:, 0], tsne_weight_c_norm[:, 1])
#for i, xy in enumerate(tsne_weight_c_norm):
#    c.annotate('%s' % (i + 1), xy=xy, textcoords='data')
plt.savefig(name_rand[:-5] + '.png')
plt.show()
