import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, num_walk=5, neg_samp=10, walk_length=80)
model.build()
model.train(mode='motif_walk',num_nodes_per_batch=400, batch_size=512)
weights = model.get_weights()
with open('BC3006.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
