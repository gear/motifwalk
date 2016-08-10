import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, num_walk=10, neg_samp=15)
model.build()
model.train(num_nodes_per_batch=200, batch_size=512)
weights = model.get_weights()
with open('random_npb200_bs512_nw10_ns15.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
