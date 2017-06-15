from motifwalk.utils import find_meta, set_dataloc, get_metadata
from motifwalk.utils.Graph import GraphContainer
from motifwalk.walks import undirected_randomwalk
from motifwalk.models.skipgram import Skipgram, ADAM
from motifwalk.utils import timer
import os
import numpy as np
from sys import argv

dataset_name = argv[1]
tnum = argv[2]
dloc = '/home/gear/Dropbox/CompletedProjects/motifwalk/data'
set_dataloc(dloc)
metadata = get_metadata()

def test(test_number):
    if '1' == test_number:
        test1()
    elif '2' == test_number:
        test2()
    elif '3' == test_number:
        test3()
    else:
        print("Test have not yet defined")

def test1():
    network = GraphContainer(find_meta(dataset_name), dloc)
    print("Generating gt graph...")
    timer()
    gt = network.get_gt_graph()
    timer(False)
    print("Creating Skipgram model...")
    timer()
    model = Skipgram(window_size=5, num_skip=2, num_nsamp=15)
    model.build(num_vertices=gt.num_vertices(), learning_rate=0.0001)
    timer(False)
    print("Generating random walk...")
    timer()
    walks, index = undirected_randomwalk(gt)
    timer(False)
    print("Start training...")
    timer()
    emb = model.train(data=walks, num_step=10, log_step=2, save_step=2)
    timer(False)
    print(type(emb))
    print(emb.shape)

def test2():
    network = GraphContainer(find_meta(dataset_name), dloc)
    print("Generating gt graph...")
    timer()
    gt = network.get_gt_graph()
    timer(False)
    print("Creating Skipgram model...")
    timer()
    model = Skipgram(window_size=5, num_skip=2, num_nsamp=15)
    model.build(num_vertices=gt.num_vertices(), learning_rate=0.001)
    timer(False)
    print("Generating random walk...")
    timer()
    walks, index = undirected_randomwalk(gt)
    timer(False)
    print("Start training...")
    timer()
    emb = model.train(data=walks, num_step=10000, log_step=2000, save_step=2)
    timer(False)
    np.save("cora.emb.npy", emb)

def test3():
    network = GraphContainer(find_meta(dataset_name), dloc)
    print("Generating gt graph...")
    timer()
    gt = network.get_gt_graph()
    timer(False)
    print("Creating Skipgram model...")
    timer()
    model = Skipgram(window_size=5, num_skip=2, num_nsamp=15)
    model.build(num_vertices=gt.num_vertices(), learning_rate=0.001, opt=ADAM)
    timer(False)
    print("Generating random walk...")
    timer()
    walks, index = undirected_randomwalk(gt)
    timer(False)
    print("Start training...")
    timer()
    emb = model.train(data=walks, num_step=1000000, log_step=2000, save_step=2)
    timer(False)
    np.save("cora.emb.npy", emb)

if __name__ == "__main__":
    test(tnum)
