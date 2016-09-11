# Abstract

Given a large complex graph, how can we have a lower dimension real-vector 
representations of vertices that preserve structural information? Recent 
advancements in graph embedding have adopted word embedding techniques 
and deep architectures to propose a feasible solution to this question. 
However, most of these former researches considers the notion of "neighborhood" 
by vertex adjacency only. In this paper, we propose a novel graph embedding 
algorithm that employs motif structures into the latent vector representation 
learning process. Our algorithm learns the graph latent representation by 
contrasting between different type of motif-biased random walk. We showed 
that our algorithm yields more accurate embedding results compared to 
other existing algorithms through various 
graph mining benchmark tasks.

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
the n-gram distribution, Skipgram maximize the local context words given a target word. The 
softmax potential function for a context word given a target word is given by:

=== Function and explainatio ===

Generally, the objective of Skipgram model is to maximize the following average log likelihood:

=== Average log likelihood ===

The straight-forward computation of the normalization factor in equation (1) is intractable
due to the size of corpus V can be arbitary large. Therefore, normalization factor estimation
technique such as hierachial softmax [hsoft] or noise contrastive estimation [nce] is needed. 
In [Skipgram], Mikholov et al.  introduced negative sampling, which can be considered as a 
simplified version of noise contrastive estimation. The objective function under negative 
sampling is given by:

=== Negative sampling objective and description ===

Under this negative sampling scheme, log likelihood maximization is replaced by classification
between true (positive) samples and negative samples. The results conducted in [skipgram, nce]
suggest that there is a strong correlation between the distribution from which we pick
the negative samples and the quality of the learned embeddings. Therefore, an appropriate 
negative sampling setting can greatly affect the runtime as well as the statistical performance
of the negative-sampling Skipgram model.

## Graph embedding

### P1: Spectral clustering, PCA

NUKUI's part.

### P2: Deepwalk, LINE, GraRep, Planetoid, Node2vec

Inspired by the power-law similarity between the vertex frequency distribution in random walk
and the word frequency distribution of English text, Peperozi et al. proposed Deepwalk algorithm
to learn latent representations of vertices in a graph [Deepwalk]. The operation of Deepwalk
is exactly similar to hierachial-softmax Skipgram, the only different is ``sentences'' are 
generated by performing random walk on the graph. However, the disadvantage of Deepwalk is 
that the algorithm ignores the graph structure. In one of the researches following Deepwalk,
Tang et al. [LINE] hinted the advantages of structure-aware graph context generation. Although
the proposed algorithm LINE by Tang et al. 

Node2vec is shady and I don't know what to say. It reports contradicting result with Deepwalk
and LINE.

## Motif in graph

### P1: Motif definition

Motif in a graph is defined as a small network portion to do stuff. Originates from biology.

### P2: Motif conductance for community definition

Motif conductant is addressed by Jure and shady Harvard guy stuff.

# Methods and experiments

## Motif-Aware context generation 

### P1: Popular motif in graph (social network and citation network) How to choose motif - triangle MC chain

### P2: Sample generation algorithm - triangle sampling

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
