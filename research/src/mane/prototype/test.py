import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
model=e.EmbeddingNet(graph=bc, skip_window=5)
model.build()
model.train(batch_size=400)
weights = model.get_weights()
with open('default_random_bs400.weights', 'wb') as f:
    p.dump(weights, f, p.HIGHEST_PROTOCOL)
