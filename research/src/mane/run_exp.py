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

import embeddings
import graph
import util

import numpy as np
from matplotlib import colors

import pickle
import logging

import os
import time

from keras.optimizers import Adam
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


from exp_config import *

g = graph.graph_from_pickle('data/{}.graph'.format(dataset_name))


# read ground truth if exists
#try:
    # {id: com} dictionary
#    correct_labels = pickle.load(open("data/{}.community", "rb"))
#except:
#    correct_labels = {i: 0 for i in range(len(g.nodes()))}


if not os.path.exists("logs"):
    os.makedirs("logs")

if not os.path.exists("save"):
    os.makedirs("save")

exp_name = "nce_{}_e{}_ed{}_ne{}_ns{}_nw{}_wl{}_ws{}_it{}_nb{}_lr{}_ci{}_adam".format(
    dataset_name, epoch, emb_dim, neg_samp, num_skip, num_walk,
    walk_length, window_size, iters, num_batches, learning_rate,
    contrast_iter)

logging.basicConfig(filename=os.path.join("logs", exp_name+".log"),
                    level=logging.INFO,
                    format="%(asctime)s %(message)s")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

if training:
    for method in methods:
        method_name = method + "_" + exp_name
        model = embeddings.EmbeddingNet(graph=g, epoch=epoch, emb_dim=emb_dim,
                                        neg_samp=neg_samp, num_skip=num_skip,
                                        num_walk=num_walk, walk_length=walk_length,
                                        window_size=window_size, iters=iters,
                                        contrast_iter=contrast_iter,
                                        learning_rate=learning_rate)
        model.build(optimizer='adam')
        logging.info("start training {}".format(method))
        start = time.time()

        if method == "rand":
            model.train(mode='random_walk', num_batches=num_batches,
                        save_dir=os.path.join("weights", method_name))
        elif method == "motif":
            model.train(mode='motif_walk', num_batches=num_batches,
                        save_dir=os.path.join("weights", method_name))
        elif method == "contrast":
            model.train_mce(pos='motif_walk', neg="random_walk",
                            num_batches=num_batches,
                            save_dir=os.path.join("weights", method_name))
        else:
            raise ValueError("methods must be in {'rand', 'motif', 'contrast'}")

        elapsed_time = time.time() - start
        logging.info('finish training: Time: {}[s]'.format(elapsed_time))
        weights = model.get_weights()
        weight_path = os.path.join("save", method_name + ".weight")
        pickle.dump(weights, open(weight_path, "wb"))


if visualize:
    topk = [c for c, _ in Counter(g.coms).most_common(topk_labels)]
    vis_nodes = [i for i, c in g.coms.items() if c in topk]
    vis_labels = [g.coms[i] for i in vis_nodes]
    color_map = ["red", "blue", "green", "gold", "purple", "oragnge", "cyan"]
    label_colors = [color_map[c%len(color_map)] for c in vis_labels]
    fig = plt.figure(figsize=(15, 45))
    fig.suptitle(exp_name, fontsize=16)
    for i, method in enumerate(methods):
        method_name = method + "_" + exp_name
        weight_path = os.path.join("save", method_name + ".weight")
        weights = pickle.load(open(weight_path, "rb"))
        embed = weights[0][vis_nodes, :]
        # Normalize
        if normalize_embed:
            embed = normalize(embed)
        embed_tsne = TSNE(learning_rate=tsne_learning_rate).fit_transform(embed)
        color_map = list(colors.cnames.keys())
        a = plt.subplot(311 + i)
        a.set_title("{} walk embedding".format(method))
        a.scatter(embed_tsne[:, 0], embed_tsne[:, 1],
                  c=label_colors, s=30)
    plt.savefig(os.path.join("save", exp_name + '.png'))
    plt.show()
