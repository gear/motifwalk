from motifwalk.utils import find_meta, set_dataloc, get_metadata
from motifwalk.utils.Graph import GraphContainer
from motifwalk.walks import undirected_randomwalk
import os
from sys import argv

dataset_name = argv[1]
tnum = argv[2]

def test(test_number):
    if '1' == test_number:
        test1()
    elif '2' == test_number:
        test2()
    else:
        print("Test have not yet defined")

def test1():
    pack = GraphContainer(find_meta(dataset_name), dloc)
    gt = pack.get_gt_graph()
    walks, index = undirected_randomwalk(gt)
    print(walks.shape)
    print(index)
    print(walks[:10])
    print(walks[-10:])


if __name__ == '__main__':
    dloc = '/home/gear/Dropbox/CompletedProjects/motifwalk/data'
    set_dataloc(dloc)
    metadata = get_metadata()
    test(tnum)
