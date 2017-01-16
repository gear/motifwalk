# Embedding and classification results

The output of related algorithms (deepwalk, node2vec, gcn, etc.) is stored
here. 

### Blogcatalog

The task for Blogcatalog graph is to classify node labels based **only on graph
structure**. Different from many other datasets with node labeling, each node 
in Blogcatalog network has multiple labels (there are 39 types of label in total).

_Note: Pay attention to the node ids. In the original scoring function used by
deepwalk, the author loads labels matrix from a different dataset (.mat file)._

- `blogcatalog.deepwalk`: 128 dimensions embeddings, walk length: 80;
number of walks: 10; skipgram window size: 10.This is the result of deepwalk algorithm.
- `blogcatalog.deepwalk2`: Rerun of `blogcatalog.deepwalk`.
- `blogcatalog.n2v`: 128 dimenstions embeddings, walk length: 80;
number of walks: 10; skipgram window size: 10, p: 0.25, q: 0.25. This is the result
of node2vec algorithm.
- `blogcatalog_deepwalk_LG.outlog`: f1 scores with different testing size. The multilabels
classifier is trained by `LogisticRegression` (sklearn) with default parameters.
- `blogcatalog_deepwalk_LGCV.outlog`: `LogisticRegressionCV` (cross-validation version).
- `blogcatalog_n2v_LG.outlog`: f1 scores for node2vec embeddings.

### Cora

Cora is a citation network with a feature vector for each nodes describing the
content of the paper. Each node has exactly one label (0-5) indicating the topic
of the paper. While embedding algorithms cannot inject the feature vector into
the embedding process, semi-supervised approaches such as planetoid and gcn can
take advantages of both graph structures and feature vectors into node classification.

- `cora.deepwalk`: 64 dimensions embeddings using deepwalk's default parameters.
