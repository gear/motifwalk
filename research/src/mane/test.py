"""Warpper for testing
"""
# Coding: utf-8
# File name: test.py
# Created: 2016-07-27
# Description:
## v0.0: File created.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import embeddings as e
import graph as g

import numpy as np
import cPickle as p

import os

from keras.optimizers import Adam
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

fb = g.graph_from_pickle('data/youtube.graph')

name_rand = 'nce_youtube_e200_ne2_ns2_nw10_wl10_ws2_adam_rand'
name_motif = 'nce_youtube_e200_ne2_ns2_nw10_wl10_ws2_adam_motif'

no_train = False

if no_train:
  pass
else:
  model_r = e.EmbeddingNet(graph=fb, epoch=200, neg_samp=2, num_skip=2, num_walk=10, walk_length=10, window_size=2)
  adam_opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model_r.build(optimizer=adam_opt)
  model_r.train(mode='random_walk', threads=8)
  weight_r = model_r._model.get_weights()
if no_train:
  pass
else:
  model_m = e.EmbeddingNet(graph=fb, epoch=200, neg_samp=2, num_skip=2, num_walk=10, walk_length=10, window_size=2)
  adam_opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model_m.build(optimizer=adam_opt)
  model_m.train(mode='motif_walk', threads=8)
  weight_m = model_m._model.get_weights()

# Save model
if not os.path.exists(name_rand+'.model'):
  with open(name_rand+'.model', 'wb') as f:
    p.dump(model_r, f, p.HIGHEST_PROTOCOL)
if not os.path.exists(name_motif+'.model'):
  with open(name_motif+'.model', 'wb') as f:
    p.dump(model_m, f, p.HIGHEST_PROTOCOL)

# Save or load data
if not os.path.exists(name_rand+'.weights'):
  with open(name_rand+'.weights', 'wb') as f:
    p.dump(weight_r, f, p.HIGHEST_PROTOCOL)
else:
  with open(name_rand+'.weights', 'rb') as f:
    weight_r = p.load(f)
if not os.path.exists(name_motif+'.weights'):
  with open(name_motif+'.weights', 'wb') as f:
    p.dump(weight_m, f, p.HIGHEST_PROTOCOL)
else:
  with open(name_motif+'.weights', 'rb') as f:
    weight_m = p.load(f)

weight_r_avg = (weight_r[0] + weight_r[1]) / 2
weight_m_avg = (weight_m[0] + weight_m[1]) / 2

# Save or load tsne
if not os.path.exists(name_rand+'.tsne'):
  tsne_weight_r_in = TSNE(learning_rate=100).fit_transform(weight_r[0])
  tsne_weight_r_out = TSNE(learning_rate=100).fit_transform(weight_r[1])
  with open(name_rand+'.tsne', 'wb') as f:
    tsne = (tsne_weight_r_in, tsne_weight_r_out)
    p.dump(tsne, f, p.HIGHEST_PROTOCOL)
else:
  with open(name_rand+'.tsne', 'rb') as f:
    tsne_weight_r_in, tsne_weight_r_out = p.load(f)

if not os.path.exists(name_motif+'.tsne'):
  tsne_weight_m_in = TSNE(learning_rate=100).fit_transform(weight_m[0])
  tsne_weight_m_out = TSNE(learning_rate=100).fit_transform(weight_m[1])
  with open(name_motif+'.tsne', 'wb') as f:
    tsne = (tsne_weight_m_in, tsne_weight_m_out)
    p.dump(tsne, f, p.HIGHEST_PROTOCOL)
else:
  with open(name_motif+'.tsne', 'rb') as f:
    tsne_weight_m_in, tsne_weight_m_out = p.load(f)

if not os.path.exists(name_rand+'_avg.tsne'):
  tsne_weight_r = TSNE(learning_rate=100).fit_transform(weight_r_avg)
  with open(name_rand+'_avg.tsne', 'wb') as f:
    p.dump(tsne_weight_r, f, p.HIGHEST_PROTOCOL)
else:
  with open(name_rand+'_avg.tsne', 'rb') as f:
    tsne_weight_r = p.load(f)

if not os.path.exists(name_motif+'_avg.tsne'):
  tsne_weight_m = TSNE(learning_rate=100).fit_transform(weight_m_avg)
  with open(name_motif+'_avg.tsne', 'wb') as f:
    p.dump(tsne_weight_m, f, p.HIGHEST_PROTOCOL)
else:
  with open(name_motif+'_avg.tsne', 'rb') as f:
    tsne_weight_m = p.load(f)

plt.figure(figsize=(10,15))
plt.subplot(321)
plt.scatter(tsne_weight_r_in[:,0], tsne_weight_r_in[:,1])
plt.subplot(322)
plt.scatter(tsne_weight_r_out[:,0], tsne_weight_r_out[:,1])
plt.subplot(323)
plt.scatter(tsne_weight_m_in[:,0], tsne_weight_m_in[:,1])
plt.subplot(324)
plt.scatter(tsne_weight_m_out[:,0], tsne_weight_m_out[:,1])
plt.subplot(325)
plt.scatter(tsne_weight_r[:,0], tsne_weight_r[:,1])
plt.subplot(326)
plt.scatter(tsne_weight_m[:,0], tsne_weight_m[:,1])
plt.show()
