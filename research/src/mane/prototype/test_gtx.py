import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, num_walk=10, neg_samp=15, walk_length=80, window_size=10)
model.build()
model.train(mode='random_walk',num_nodes_per_batch=200, batch_size=1000)
weights = model.get_weights()
with open('BC3019.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
