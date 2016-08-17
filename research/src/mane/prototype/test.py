import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, num_walk=20, neg_samp=15, walk_length=80)
model.build()
model.train(mode='motif_walk',num_nodes_per_batch=100, batch_size=128)
weights = model.get_weights()
with open('BC3012.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
