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

- Motif-aware graph context generation process emphasizes the motif structure, which
is a strong indicator of communities [Jure motif, Harvard motif]. Tang et al. and Grover et al. also
have similar idea of manipulating graph context generation in their researches [LINE, nove2vec].

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

=== Function and explaination ===

Generally, the objective of Skipgram model is to maximize the following average log likelihood:

=== Average log likelihood ===

The straight-forward computation of the normalization factor in equation (1) is intractable
due to the size of corpus V can be arbitary large. Therefore, normalization factor estimation
technique such as hierachial softmax [hsoft] or noise contrastive estimation [nce] is needed. 
In [Skipgram], Mikholov et al. introduced negative sampling, which can be considered as a 
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

=== + Mention two phases of context generation ===

=== Figure describe skipgram and context generation ===

Inspired by the power-law similarity between the vertex frequency distribution in random walk
and the word frequency distribution of English text, Peperozi et al. proposed Deepwalk algorithm
to learn latent representations of vertices in a graph [Deepwalk]. The operation of Deepwalk
is exactly similar to hierachial-softmax Skipgram, the only different is "sentences" are 
generated by performing random walk on the graph. Despite simplicity, the disadvantage of Deepwalk is 
that the algorithm ignores the graph structure. In one of the researches following Deepwalk,
Tang et al. [LINE] hinted the advantages of structure-aware graph context generation in the
experimental result of citation network. The algorithm LINE, proposed by Tang et al., employs 
a second order proximity scheme
for graph context generation [LINE]. In our research, we noticed that LINE might be out performed
by Deepwalk and other algorithms [Deepwalk, GraRep, Planetoid, Node2vec], but it has better 
performance in citation graphs. The credit for such competitive performance can be given to
the second order proximity scheme, which is stark similar to the bipartite motif.
From this observation, we hypothesize that a graph embedding
quality can be improved by using statistically significant motif within the given graph.
In the most recent work in graph embedding, Grover et al. proposed Node2vec algorithm [Node2vec],
which controls the graph context generation by a biased random walk. This biased random
walk idea is similar to our graph context generation here. While both node2vec and our algorithm
performs biasd random walk, the objectives are different. Node2vec aims to emphasize "important"
vertices, while our algorithm aims to emphasize the local motif structure.

There is another appreciable graph embedding framework named Planetoid proposed by Yang et al. 
[Planetoid]. In their framework, Deepwalk algorithm with NCE-Skipgram is employed to learn 
the graph embeddings. In addition, the framework also takes graph labels training sets into
the latent vectors learning procedure. Planetoid is empirically proven to outperform other
embedding algorithms. Although Planetoid has high performance, we consider it belongs to another
category since the algorithm requires not only the graph itself, but also the ground-truth 
graph labels input. The discussion of combining Planetoid framework and our embedding algorithm 
is in Section 5 of this paper.

=== Figure describe LINE ===

=== Figure describe node2vec ===

=== Figure describe motif community ===

## Motif in graph

### P1: Motif definition

Graph motifs (also network motifs) are defined to be recurring and regulation patterns in a
complex graph [UriAlonNetworkMotifs]. Graph motifs concept was first mentioned by Milo et al. 
[MiloNetworMotifsBlock], and it has been an important branch of research in real-world graph 
models ever since. Although there was debation in the usage and functionality of network motifs 
[MasoudiReview], recent researches suggested that graph motifs withold the structural 
information of the graph and they are useful especially in community detection [JureConductance,
HarvardConductance, DeepgraphKernel]. Due to the computational complexity of graph isomorphism
task, the number of vertices for each motif definition is often less than 4. Figure XXX presents
directed and undirected graph motifs.

=== Figure describe motifs ===

### P2: Motif conductance, motif detection, motif building blocks

The sentence "Friend of friend is friend" holds true in life. Expectedly, this sentence is also 
true for social network. Social network studies [Social network studies] showed the correlation
between the undirected triangle motif (Fig XXX) and the social community. On the other hand, we
also observe that the social networks that involve human being friend with each other have higher
triangle count compared to other type of networks such as citation networks. Similarly, the bipartite
motif (Fig XXX) is found much more commonly in citation networks than other networks. Because of
such observation, researchers have proposed motif-aware techniques for community detection and
clustering [motif community detection]. MORE HERE LATER.

=== Formal definition of graph

=== Formal definition of motif

=== + Mention motif size definition

In practice, real-world networks often have a small number of motif types, which researchers 
refered as the building blocks of the network [MotifBuildingBlock]. In order to discover these
building blocks, there were many proposed algorithms to detect the statistically significant 
motifs in a network [MotifdetectAlgo]. 

# Methods and experiments

In this section, we present our algorithm in detail. Although there are many different motif
structure in graph, we only consider undirected triangle motif (Fig XXX), undirected wedge 
motif (Fig XXX), and undirected bipartite motif (Fig XXX) in this paper for the sake of
simplicity. Other motif and directed graph extension is discussed in Section 5. 

## Motif-Aware context generation 

### P1: Popular motif in graph (social network and citation network) How to choose motif - triangle MC chain

We use negative sampling Skipgram (NEG-Skipgram) model as the base model in our paper. 
As mentioned in the previous section, graph embedding _is_ word embedding, the difference 
is the "sentences" must be generated by a random process in graph embedding. 
For the positive sampling phase, we define a motif-biased random walk, or motif walk for 
short, which generates the graph context biased toward the choosen motif. Our motif walk
is a MC Markov chain [MCMC], whose number of states equals the motif size. At each 
step along the motif walk, an adjacent vertex is picked by the rejection sampling procedure,
the rejection probability depends on the state of the Markov chain. The general algorithm
for an arbitary motif is given in algorithim XXX.

=== MCMC formal definition ===

=== Algorithm based on an arbitary Markov Chain ===.

### P2: Sample generation algorithm - triangle sampling

The motif detection procedure for each graph is out of the scope of this paper. Therefore, in
this work, we demonstrate the most popular motifs found in social networks and citation networks only. 
Figure XXX and algorithm XXX describle the undirected triangle motif walk. This triangle motif is the most
common motif in social networks, especially within a community. In addition, undirected biparite motif
is represented in Figure XXX, and undirected wedge motif is presented in Figure XXX.

=== MCMC for triangle walk ===

=== Algorithm for triangle motif rejection sampling ===

## Negative sampling

Our negative sampling also based on motif   

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
