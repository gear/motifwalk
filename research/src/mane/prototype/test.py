import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/youtube.graph')
model=e.EmbeddingNet(graph=bc, num_walk=1, neg_samp=10, emb_dim=8)
model.build()
model.train(mode='motif_walk', num_nodes_per_batch=400, batch_size=1024)
weights = model.get_weights()
with open('youtube_motif_npb400_bs1024_nw10_ns10.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
