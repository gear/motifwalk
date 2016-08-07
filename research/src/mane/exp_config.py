# coding: utf-8


training = True
visualize = True

methods = ["rand", "motif", "contrast"]
dataset_name = "karate"
index_cols = True
epoch = 1
emb_dim = 100
neg_samp = 3
num_skip = 3
num_walk = 2
walk_length = 10
window_size = 3
iters = 1
num_batches = 5000

# setting for visualization
normalize_embed = True
tsne_learning_rate = 200
