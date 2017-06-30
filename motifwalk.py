import os
import argparse
import numpy as np

from motifwalk.utils import find_meta, set_dataloc, get_metadata, timer
from motifwalk.utils.Graph import GraphContainer
from motifwalk.walks import undirected_randomwalk, undirected_rw_kernel, \
                            ParallelWalkPimp
from motifwalk.models.skipgram import Skipgram, ADAM, EdgeEmbedding

from motifwalk.motifs import all_u3, all_3, all_u4, all_4
from motifwalk.motifs.analysis import construct_motif_graph, filter_isolated

triangle = all_u3[0]


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str,
                    help="Dataset name (e.g. cora).")
parser.add_argument("-w", "--walk_type", type=str,
                    help="Type of the walk on graph (e.g. random).",
                    default="undirected")
parser.add_argument("-mt", "--motif", type=str,
                    help="Type of motif to run.",
                    default="triangle")
parser.add_argument("-l", "--walk_length", type=int,
                    help="Length of random walk starting from each node.",
                    default=80)
parser.add_argument("-k", "--emb_dim", type=int,
                    help="Embedding dimensionality.", default=16)
parser.add_argument("-t", "--num_step", type=int,
                    help="Number of step to train the embedding.",
                    default=10000)
parser.add_argument("-nw", "--num_walk", type=int,
                    help="Number of random walk per graph node. Also number of \
                    parallel processes if parallel run is used.",
                    default=10)
parser.add_argument("-ep", "--enable_parallel", type=bool,
                    help="Enable parallel random walk.", default=True)
parser.add_argument("-b", "--batch_size", type=int,
                    help="Batch size.")
parser.add_argument("-m", "--model", type=str,
                    help="Embedding model.", default="skipgram")
parser.add_argument("-ws", "--window_size", type=int,
                    help="Skipgram window size.", default=5)
parser.add_argument("-nn", "--num_neg", type=int,
                    help="Number of negative samples each contex.", default=15)
parser.add_argument("-ns", "--num_skip", type=int,
                    help="Number of skips per window.", default=2)
parser.add_argument("-lr", "--learning_rate", type=float,
                    help="The learning rate.", default=0.05)
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

    print("Creating {} model...".format(args.model))
    timer()
    model = None
    modelm = None
    if "skipgram" == args.model.lower():
        model = Skipgram(window_size=args.window_size, num_skip=args.num_skip,
                         num_nsamp=args.num_neg, name=args.dataset)
    elif "skipgram_motif" == args.model.lower():
        model = Skipgram(window_size=args.window_size, num_skip=args.num_skip,
                         num_nsamp=args.num_neg, name=args.dataset)
        modelm = Skipgram(window_size=args.window_size, num_skip=args.num_skip,
                         num_nsamp=args.num_neg, name=args.dataset+"m")
    elif "edge_embedding" == args.model.lower():
        model = EdgeEmbedding(num_nsamp=args.num_neg, name=args.dataset)
    elif "gcn" == args.model.lower():
        print ("TODO")
    elif "sc" == args.model.lower():
        print ("TODO")
    else:
        print("Unknown embedding model.")
    assert model is not None
    if modelm is not None:
        model.build(num_vertices=gt.num_vertices(), emb_dim=args.emb_dim//2,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                regw=args.reg_strength, device=args.device)
    else:
        model.build(num_vertices=gt.num_vertices(), emb_dim=args.emb_dim,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                regw=args.reg_strength, device=args.device)
    timer(False)

    print("Generating walks...")
    timer()
    walks = None
    mwalks = None
    if "undirected" == args.walk_type and not args.enable_parallel:
        walks, _ = undirected_randomwalk(gt, walk_length=args.walk_length,
                                             num_walk=args.num_walk)
        timer(False)
        if modelm is not None:
            print("Generating motifwalk...")
            timer()
            assert len(args.motif)
            motif = eval(args.motif)
            motif_graph = construct_motif_graph(graph, motif)
            motif_view = filter_isolated(motif_graph)
            mwalks, _ = undirected_randomwalk(motif_view,
                                            walk_length=args.walk_length,
                                            num_walk=args.num_walk)
    elif "undirected" == args.walk_type and args.enable_parallel:
        pwalker = ParallelWalkPimp(gt, undirected_rw_kernel,
                                   args=(args.walk_length,),
                                   num_proc=args.num_walk)
        walks = pwalker.run()
        timer(False)
        if modelm is not None:
            print("Generating motifwalk...")
            timer()
            assert len(args.motif)
            motif = eval(args.motif)
            motif_graph = construct_motif_graph(graph, motif)
            motif_view = filter_isolated(motif_graph)
            pmwalker = ParallelWalkPimp(motif_view, undirected_rw_kernel,
                                       args=(args.walk_length,),
                                       num_proc=args.num_walk)
            mwalks = pmwalker.run()
    elif "edges" == args.walk_type:
        walks = graph.get_graph()  # walks here is the networkx version
    else:
        print("TODO")
    assert walks is not None
    timer(False)

    print("Start training ...")
    timer()
    emb = model.train(data=walks, num_step=args.num_step,
                      log_step=args.log_step, save_step=args.save_step,
                      learning_rate=args.learning_rate)
    memb = None
    if modelm is not None:
        print("Start building and training for motif model...")
        modelm.build(num_vertices=gt.num_vertices(), emb_dim=args.emb_dim//2,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    regw=args.reg_strength, device=args.device, init_emb=emb)
        memb = modelm.train(data=mwalks, num_step=args.num_step,
                            log_step=args.log_step, save_step=args.save_step,
                            learning_rate=args.learning_rate)
    timer(False)

    from time import time
    uid = str(time())
    np.save(args.save_loc+"{}_{}.emb".format(args.dataset, uid), emb)
    if memb is not None:
        np.save(args.save_loc+"{}_{}.memb".format(args.dataset, uid), memb)

    with open(args.save_loc+"{}_{}.info".format(args.dataset, uid),
              "w") as infofile:
        infofile.write(uid + '\n')
        args_dict = vars(args)
        for key, val in args_dict.items():
            infofile.write("{}: {}\n".format(key,val))

if __name__ == "__main__":
    main()
