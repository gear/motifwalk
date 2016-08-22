import graph as g
import embeddings as e
import pickle as p

bc = g.graph_from_pickle('../data/blogcatalog3.graph')
embDim = 128

posFunc = 'random_walk'
posArgs = {'walk_length': 80, 'start_node': None, 'rand_seed': None,
            'reset': 0.0, 'walk_bias': 0.99, 'isNeg': False}
negFunc = 'unigram'
negArgs = {'walk_length': 80, 'start_node': None, 'rand_seed': None,
            'reset': 0.0, 'walk_bias': 0.75, 'isNeg': True}
ep = 1
negSamp = 15
numSkip = 5
numWalk = 10
walkLength = 80
windowSize = 10
walkPerBatch = 200
batchSize = 2000
vb = 1

model=e.EmbeddingNet(graph=bc, emb_dim=embDim)
model.build()
model.train(pos_func=posFunc, neg_func=negFunc, epoch=ep,
            neg_samp=negSamp, num_skip=numSkip, num_walk=numWalk,
            walk_length=walkLength, window_size=windowSize,
            walk_per_batch=walkPerBatch, batch_size=batchSize, 
            verbose=vb)
weights = model.get_weights()
with open('embeddings/BC3026.weights', 'wb') as f:
  p.dump(weights, f, p.HIGHEST_PROTOCOL)
