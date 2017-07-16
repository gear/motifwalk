import os
import argparse
import numpy as np
import networkx as nx

from motifwalk.utils import find_meta, set_dataloc, get_metadata, timer
from motifwalk.utils.Graph import GraphContainer
from motifwalk.walks import undirected_randomwalk, undirected_rw_kernel, \
                            ParallelWalkPimp
from motifwalk.models.skipgram import Skipgram, ADAM, MotifEmbedding

from motifwalk.motifs import all_u3, all_3, all_u4, all_4
from motifwalk.motifs.analysis import construct_motif_graph, filter_isolated

triangle = all_u3[0]


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str,
                    help="Dataset name (e.g. cora).")
parser.add_argument("-mt", "--motif", type=str,
                    help="Type of motif to run.",
                    default="triangle")
parser.add_argument("-ac", "--anchors", type=str,
                    help="Set of anchor node id. (e.g. {1,2})", default=None)
parser.add_argument("-l", "--walk_length", type=int,
                    help="Length of random walk starting from each node.",
                    default=80)
parser.add_argument("-k", "--emb_dim", type=int,
                    help="Embedding dimensionality.", default=16)
parser.add_argument("-t", "--num_step", type=int,
                    help="Number of step to finetune the embedding.",
                    default=100000)
parser.add_argument("-tm", "--num_mstep", type=int,
                    help="Number of steps for motif training.",
                    default=50000)
parser.add_argument("-nw", "--num_walk", type=int,
                    help="Number of random walk per graph node. Also number of \
                    parallel processes if parallel run is used.",
                    default=10)
parser.add_argument("-ep", "--enable_parallel", type=bool,
                    help="Enable parallel random walk.", default=True)
parser.add_argument("-b", "--batch_size", type=int,
                    help="Batch size.")
parser.add_argument("-ws", "--window_size", type=int,
                    help="Skipgram window size.", default=5)
parser.add_argument("-nn", "--num_neg", type=int,
                    help="Number of negative samples each contex.", default=15)
parser.add_argument("-ns", "--num_skip", type=int,
                    help="Number of skips per window.", default=2)
parser.add_argument("-lr", "--learning_rate", type=float,
                    help="The learning rate.", default=0.05)
parser.add_argument("-fr", "--finetune_rate", type=float,
                    help="Fine-tunning rate for final embedding.", default=0.01)
parser.add_argument("--log_step", type=int,
                    help="Number of step to report average loss.", default=2000)
parser.add_argument("--save_step", type=int,
                    help="Number of step to save the model.", default=8000)
parser.add_argument("--device", type=str,
                    help="Select device to run the model on using TF format",
                    default="/cpu:0")
parser.add_argument("--reg_strength", type=float,
                    help="Regularization strength for embedding matrix",
                    default=0.8)
parser.add_argument("--save_loc", type=str, help="Embedding save location.",
                    default="./")

def main():
    args = parser.parse_args()
    dloc = '/home/gear/Dropbox/CompletedProjects/motifwalk/data'
    set_dataloc(dloc)
    metadata = get_metadata()

    graph = GraphContainer(find_meta(args.dataset), dloc)
    print("Generating gt graph...")
    timer()
    gt = graph.get_gt_graph()
    timer(False)

    print("Creating MotifEmbedding model...")
    timer()
    model = None
    modelm = None
    model = MotifEmbedding(window_size=args.window_size, num_skip=args.num_skip,
                           num_nsamp=args.num_neg, name=args.dataset)
    model.build(num_vertices=gt.num_vertices(), emb_dim=args.emb_dim,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                regw=args.reg_strength, device=args.device)

    print("Generating motifwalk...")
    timer()
    assert len(args.motif)
    motif = eval(args.motif)  # TODO: dont use eval
    print(motif)
    if (args.anchors is not None):
        motif.anchors = eval(args.anchors)   # TODO: avoid eval
        print(motif.anchors)
    motif_graph = construct_motif_graph(graph, motif)
    motif_view = filter_isolated(motif_graph)
    def to_int_tuple(t):
        t = tuple(t)
        return (int(t[0]), int(t[1]))
    all_motif_edges = [*map(to_int_tuple, motif_view.edges())]
    print(len(all_motif_edges))
    motif_nx_graph = nx.Graph()
    motif_nx_graph.add_edges_from(all_motif_edges)
    timer(False)

    print("Create random walk context...")
    timer()
    pwalker = ParallelWalkPimp(gt, undirected_rw_kernel,
                               args=(args.walk_length,),
                               num_proc=args.num_walk)
    walks = pwalker.run()
    timer(False)
    print("Training with motif...")
    timer()
    emb = model.train(data=walks, nxg=motif_nx_graph, num_step=args.num_step,
                num_mstep=args.num_mstep, log_step=args.log_step,
                save_step=args.save_step,
                learning_rate=args.learning_rate,
                finetune_rate=args.finetune_rate)
    timer(False)
    from time import time
    uid = str(time())
    np.save(args.save_loc+"{}_{}.emb".format(args.dataset, uid), emb)

    with open(args.save_loc+"{}_{}.info".format(args.dataset, uid),
              "w") as infofile:
        infofile.write(uid + '\n')
        args_dict = vars(args)
        infofile.write("Motif edge init.\n")
        for key, val in args_dict.items():
            infofile.write("{}: {}\n".format(key,val))

if __name__ == "__main__":
    main()
