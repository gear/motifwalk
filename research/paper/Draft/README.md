# Abstract

# Introduction

### P1: Context

The graph (or network) data model is an useful tool for a wide 
range of disciplines and it is important to have a low dimensionality 
representation of a complex graph. A graph, in its simplest form, is a 
set of vertices (nodes) and edges connecting them. 
Being simple and expresive, the graph-theoric approach has been applied
in many scientic fields and proven to be a powerful in structure discovery. For
example, the study conducted by [MP Van Den Heuvel] suggests that brain functional-
connectivity network provides insights about the human brain's organization. In the
same way, other complex structures such as social networks or molecule interaction 
networks are also studied under the graph model. However, 
the graph analysis process becomes challenging as the scientic problem deals with
complex data. In such case, the graph usually contains several thousands to millions
of vertices. Therefore, it is deseriable to have a latent representation of the
graph with a lower dimensionality while retaining the structural properties.
A *meaningful* latent representation of a graph can benefit researchers
in many ways. For instance, instead of relying on only graph algorithms to analyze
a given structure, researchers can also apply other machine learnings algorithms on
the latent representation to make predictions on the data.
In summary, graph embedding techniques promise a feasible solutions to many graph analysis tasks.

### P2: Need

Finding a latent representation can be a challenging task.
Traditionally, eigenvalues-based techniques such as spectral clustering, PCA, or CCA
provide a good projection (embedding) from the adjacency matrix representation to real vector
representation. Nevertheless, these methods are impractical when the graph
is large [Deepwalk, PCA]. In 2014, Peperozi et al. proposed Deepwalk [Deepwalk] -
a skipgram-based [Skipgram] algorithm for graph embedding. Deepwalk is a fast and
scalable approach to *learn* the latent representation of some given graph. By treating
a random series of vertices as if it is a sentence of words, Deepwalk utilizes Skipgram model to
learn latent real-vector representations. Succeeding algorithms to Deepwalk such as LINE [LINE], 
platenoid [platenoid], and node2vec [node2vec] further improve embedding quality by 
taking network structures or community labels into the embedding process. However,
all of the aforementioned algorithms lack motif structure consideration in their
normalization factor estimation process.

### P3: Introduce skipgram with negative sampling

In this work, we propose an algorithm which controls both graph context and
negative samples generation. The motif-aware context generation aims to emphasize
vertices that are in the same motif community structure, while inverse-motif negative
samples generation further improves the normalization factor estimation, which is
intrinsic to Skipgram model. Generally, our algorithm has two following advantages:

- Motif-aware positive samples generation process emphasizes the motif structure, which
is a strong indicator of communities [Jure motif, Harvard motif]. Tang et al. and Grover et al. also
have similar idea mentioned in their researches [LINE, nove2vec].

- Instead of using distorted degree distribution or uniform distribution as negative
sampling, we use another motif-biased random walk as a negative sample generator. Intuitively,
this negative sampling procedure generates vertices that are close to the target community,
but does not belong to the motif structure, which also an indicator for overlapping communities.

The algorithm implementation and experimental results are available on Github. [Footnote Link]

### P4: Structure of the document

The remaining of this paper is divided into 4 parts. The related works and background
are provided in Section 2. Our implementation detail is presented 
in Section 3. The result discussion and conclusion are Section 4 and 5 consecutively.

# Related work

## Skipgram model and negative sampling

### P1: Natural Language Processing model (Softmax, unigram, skipgram)

Representation learning has been a key role in the current success of machine learning
algorithms [Bengio Review]. In the context of natural language processing (NLP), representation
learning becomes even more important as the data has large dimension. To address the dimensionality 
problem in NLP, Mikholov et al. have proposed the Skipgram model [Skipgram]. Instead of maximizing
the n-gram distribution, Skipgram maximize the local context words given a target word. The log 
likelihood function is given by:

$$ \log $$

### P2: Normalization factor estimation with negative sampling, negative sampling loss function

## Graph embedding

### P1: Spectral clustering, PCA

### P2: Deepwalk, LINE, GraRep, Planetoid

### P3: node2vec

## Motif in graph

### P1: Motif definition

### P2: Motif conductance for community definition

# Methods and experiments

## Motif-Aware context generation 

### P1: Popular motif in graph (social network and citation network)

### P2: Sample generation algorithm

## Negative sampling

### P1: Inverse motif generation

### P2: Negative sample generation algorithm

## Motif-Aware Graph Embedding Model

### P1: Loss function definition (learning ratio)

### P2: Distance function

## Experiments design

### P1: Multilabel classification (mention dataset)

### P2: Link prediction (mention dataset)

# Results and discussion

## Multilabel classification result

### P1: Result

### P2: Discussion

## Link prediction

### P1: Result

### P2: Discussion

## Discussion
