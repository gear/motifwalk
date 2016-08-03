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
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

fb = g.graph_from_pickle('data/egonets.graph')

name_rand = 'nce_egonets_e10_ne15_ns2_nw5_wl10_ws3_it1_adam_rand'
name_motif = 'nce_egonets_e10_ne15_ns2_nw5_wl10_ws3_it1_adam_motif'
name_contrast = 'nce_egonets_e10_ne15_ns2_nw5_wl20_ws3_it1_adam_contrast'

rand_train = False
motif_train = False
contrast_train = True

save_rand = False
save_motif = False
save_contrast = True

if not rand_train:
  pass
else:
  model_r = e.EmbeddingNet(graph=fb, epoch=10, emb_dim=200, neg_samp=15, 
                           num_skip=2, num_walk=5, walk_length=10, 
                           window_size=3, iters=1.0)
  adam_opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model_r.build(optimizer='adam')
  model_r.train(mode='random_walk')
  weight_r = model_r._model.get_weights()

if not motif_train:
  pass
else:
  model_m = e.EmbeddingNet(graph=fb, epoch=10, emb_dim=200, neg_samp=15,
                           num_skip=2, num_walk=5, walk_length=10, 
                           window_size=3, iters=1.0)
  adam_opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model_m.build(optimizer='adam')
  model_m.train(mode='motif_walk')
  weight_m = model_m._model.get_weights()

if not contrast_train:
  pass
else:
  model_c = e.EmbeddingNet(graph=fb, epoch=10, emb_dim=200, neg_samp=15,
                           num_skip=2, num_walk=5, walk_length=20,
                           window_size=3, iters=1.0) # reset default at 0.3
  model_c.build(optimizer='adam')
  model_c.train_mce()
  weight_c = model_c._model.get_weights()

# Save or load data
if not os.path.exists(name_rand+'.weights') and load_rand:
  with open(name_rand+'.weights', 'wb') as f:
    p.dump(weight_r, f, p.HIGHEST_PROTOCOL)

if not os.path.exists(name_motif+'.weights') and load_motif:
  with open(name_motif+'.weights', 'wb') as f:
    p.dump(weight_m, f, p.HIGHEST_PROTOCOL)

if not os.path.exists(name_contrast+'.weights') and load_contrast:
  with open(name_contrast+'.weights', 'wb') as f:
    p.dump(weight_c, f, p.HIGHEST_PROTOCOL)

# Normalize
weight_r_avg = normalize(weight_r[0])
weight_m_avg = normalize(weight_m[0])
weight_c_avg = normalize(weight_c[0])

# Save or load tsne
if not os.path.exists(name_rand+'.tsne') and save_rand:
  tsne_weight_r_in = TSNE(learning_rate=100).fit_transform(weight_r[0])
  tsne_weight_r_out = TSNE(learning_rate=100).fit_transform(weight_r[1])
  with open(name_rand+'.tsne', 'wb') as f:
    tsne = (tsne_weight_r_in, tsne_weight_r_out)
    p.dump(tsne, f, p.HIGHEST_PROTOCOL)

if not os.path.exists(name_motif+'.tsne') and save_motif:
  tsne_weight_m_in = TSNE(learning_rate=100).fit_transform(weight_m[0])
  tsne_weight_m_out = TSNE(learning_rate=100).fit_transform(weight_m[1])
  with open(name_motif+'.tsne', 'wb') as f:
    tsne = (tsne_weight_m_in, tsne_weight_m_out)
    p.dump(tsne, f, p.HIGHEST_PROTOCOL)

if not os.path.exists(name_contrast+'_avg.tsne') and save_contrast:
  tsne_weight_c = TSNE(learning_rate=100).fit_transform(weight_c[0])
  tsne_weight_c_norm = TSNE(learning_rate=100).fit_transform(weight_c_avg)
  with open(name_contrast+'_norm.tsne', 'wb') as f:
    p.dump(tsne_weight_c, f, p.HIGHEST_PROTOCOL)

#fig = plt.figure(figsize=(10,15))
#fig.suptitle(name_rand[:-5], fontsize=16)
#a=plt.subplot(321)
#a.set_title("Random walk in embedding")
#a.scatter(tsne_weight_r_in[:,0], tsne_weight_r_in[:,1])
#b=plt.subplot(322)
#b.set_title("Random walk out embedding")
#b.scatter(tsne_weight_r_out[:,0], tsne_weight_r_out[:,1])
#c=plt.subplot(323)
#c.set_title("Motif walk in embedding")
#c.scatter(tsne_weight_m_in[:,0], tsne_weight_m_in[:,1])
#d=plt.subplot(324)
#d.set_title("Motif walk out embedding")
#d.scatter(tsne_weight_m_out[:,0], tsne_weight_m_out[:,1])
#e=plt.subplot(325)
#e.set_title("Random walk average embedding")
#e.scatter(tsne_weight_r[:,0], tsne_weight_r[:,1])
#f=plt.subplot(326)
#f.set_title("Motif walk average embedding")
#f.scatter(tsne_weight_m[:,0], tsne_weight_m[:,1])
#plt.savefig(name_rand[:-5]+'.png')
#plt.show()

fig = plt.figure(figsize=(5,10))
fig.suptitle(name_contrast, fontsize=16)
a = plt.subplot(121)
a.set_title("Un-normalized embedding")
a.scatter(tsne_weight_c[:,0], tsne_weight_c[:,1])
b = plt.subplot(122)
b.set_title("Normalized embedding")
b.scatter(tsne_weight_c_norm[:,0], tsne_weight_c_norm[:,1])
plt.savefig(name_contrast+'.png')
plt.show()
