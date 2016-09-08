# Abstract

# Introduction

## Paragraph 1: Context

This paragraph should provides:
- Some context to orient those readers who are less familiar 
with graph and graph embedding.

### Main idea: Graph is important in many fields.

The graph (or network) data model is an useful tool for a wide 
range of disciplines and it is important to have a low dimensionality 
representation of a complex graph.

### Supporting 1: What is graph (or network).

Graph, in its simplest form, is a set of vertices (nodes) and edges
connecting them. 

### Supporting 2: Many discoveries are from analizing graph structures.

Because of its representation power, the graph-theoric approach has been applied
in many scientic fields and proven to be a powerful in structure discovery. For
example, the study conducted by [MP Van Den Heuvel] suggests that brain functional-
connectivity network provides insights about the human brain's organization. In the
same way, other complex structures such as social networks or molecule interaction 
networks are also studied under the graph model.

### Supporting 3: What is complex network? Many system must be modeled as a complex network.

The graph analysis process becomes challenging as the scientic problem deals with
complex data.

### Supporting 4: Dimensionality problem, mapping solution (PCA, CCA, embeddings).

complex data. In such case, the graph usually contains several thousands to millions
of nodes. Therefore, it is deseriable to have a latent representation of the
graph with a lower dimensionality while retaining the structural properties.

### Supporting 5: Usefulness of embedding techniques.

A *meaningful* latent representation of a graph can benefit researchers
in many ways. For instance, instead of relying on only graph algorithms to analyze
a given structure, researchers can also apply other machine learnings algorithms on
the latent representation to make predictions on the data.

### Conclusion: Graph model is important and learning latent representation can help solving many problems.

## Paragraph 2: Need (maybe skip for now)

Traditionally, eigenvalues-based techniques such as spectral clustering, PCA, or CCA
provide a good projection (embedding) from the adjacency matrix representation to real vector
representation. However, these methods are known to be impractical when the graph
is large [Deepwalk, PCA]. In 2014, Peperozi et al. proposed Deepwalk [Deepwalk], 
a skipgram-based [Skipgram] algorithm for graph embedding. Deepwalk provides a
fast and scalable 
Some of the "need" is mentioned in paragraph 1.

## Paragraph 3: Introduce skipgram with negative sampling

### Main idea: Careful negative sample selection algorithm improves the result.

### Supporting 1: Skipgram model depends on normalization factor estimation.

### Supporting 2: Common use is negative sampling which is a simplified version of NCE.

### Supporting 3: Control positive walk similar to node2vec proves to be useful.

### Supporting 4: Control negative sample generation will further improve the work.

### Supporting 5: Use motif and negative motif as basis for generating these.

## Paragraph 4: Structure of the document

### Main idea: The document is divided into 4 parts.

### Supporting 1: Methods and data.

### Supporting 2: Results.

### Supporting 3: Discussion.

### Supporting 4: Related works.

### Supporting 5: Discussion.

# Methods

## Subsection 1: Probabilistic model

## Subsection 2: Positive sample generation

## Subsection 3: Negative sample generation

## Subsection 4: Implementation details
