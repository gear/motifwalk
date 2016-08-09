import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, neg_samp=10)
model.build(optimizer='sgd')
model.train(batch_size=400)
weights = model.get_weights()
with open('random_bs400_ns10.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
