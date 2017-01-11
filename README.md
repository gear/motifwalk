# MotifWalk: Network local structural representation embedding.

Murata Laboratory - Tokyo Institute of Technology.

## Introduction

MotifWalk is a novel network embedding algorithm that not only encode the notion of proximity between nodes in a network, but also its local structure. By utilizing the local structures known as motif, we have successfully developed MotifWalk to learn meaningful dense vector representation of nodes in a complex network. MotifWalk is also fast and scalable. MotifWalk is written in Python and use TensorFlow framework for computation tasks.

## Repository structure

This repository is used to store research log and source code of MotifWalk.

1. __research__: Stores research log, figures, other graph embedding algorithms' source code. 
2. __src__: MotifWalk source code.
3. __data__: Store processed and raw dataset used in the research.
3. __docs__: Documentation.
4. __build__: Builds for different platforms.
5. __bin__: Some prebuilt versions.

## Environment

MotifWalk is developed using Python 3.5.2. Additional packages:

- NetworkX 1.11
- Tensorflow 0.10.0rc0
- Sklearn \& Scipy 0.18.1
- Numpy 1.11.2
- Pickle compatible formats: `['1.0', '1.1', '1.2', '1.3', '2.0', '3.0', '4.0']`

## Datasets

The dataset is stored as a pickle with `.data` extension. The dataset is a
dictionary with keys (e.g. 'NXGraph' or 'Labels') and corresponding values.
The graph data is stored as NetworkX graph (directed and undirected). The labels
are stored as numpy list (single label) or list of list matrix (multiple labels).
Some dataset also has features vector for each node, this feature vector is stored
as a compressed sparse row matrix. In all dataset, the node ids are integers and
it corresponse to the indices of the first axis of features matrices and labels.

## Quick tutorial

Read docs for more information. Here I only provide a quick example code for starting MotifWalk.


## Installation

## Citing

## Misc
