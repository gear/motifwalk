# Graph datasets

This folder contains pickled graph data. __raw__ folder contains raw data file from its original source.

## Social networks

### Facebook personal network: Egonets
  1. Source: [https://snap.stanford.edu/data/egonets-Facebook.html](Stanford SNAP - Facebook Egonets).
  2. Format: Raw: text edge list. Pickle: Dictionary of adjacency list.
  3. Nodes: 4,039. Edges: 88,234. Triangles: 1,612,010. Fraction of closed triangles: 0.2647. Diameter: 8.
  4. File name: facebook\_egonets.graph (cPickle -> dict).
  5. NOTE: Undirected graph with unique edge list. (i.e. (1,2) means (1,2) and (2,1), in adj list of 2 won't have 1).

### Amazon product network with ground truth
  1. Source: [https://snap.stanford.edu/data/com-Amazon.html](Stanford SNAP - Amazon product co-purchasing network and ground-truth communities).
  2. Format: Raw: text edge list. Pickle: Dictionary of adjacency list.
  3. Nodes: 334,863. Edges: 925,872. Triangles: 667,129. Fraction of closed traiangles: 0.07925. Diameter: 44.
  4. File name: amazon\_copurchase.graph (cPickle -> dict). amazon\_copurchase\_communities.labels (cPickle -> dict).
  5. NODE: Undirected graph with unique edge list.
