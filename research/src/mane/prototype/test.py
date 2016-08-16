import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, num_walk=5, neg_samp=15)
model.build()
model.train(mode='motif_walk',num_nodes_per_batch=200, batch_size=1024)
weights = model.get_weights()
with open('youtube_motif_npb400_bs1024_nw10_ns10.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
