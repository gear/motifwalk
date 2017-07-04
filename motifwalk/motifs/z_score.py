from motifwalk.utils import find_meta, set_dataloc, get_metadata, timer
from motifwalk.utils.Graph import GraphContainer
from motifwalk.motifs import all_4, all_3, all_u4, all_u3

from graph_tool.all import motif_significance, motifs, Graph

from random import shuffle

import argparse

default_dloc = "/home/gear/Dropbox/CompletedProjects/motifwalk/data"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, help="Name of network.")
parser.add_argument("-f", "--output", type=str, help="Write to file name.")
parser.add_argument("-k", "--motif_size", type=int, help="Size of motifs.")
parser.add_argument("-n", "--num_shuffles", type=int, default=10,
                    help="Number of network shuffles to compute z-score.")
parser.add_argument("--dloc", type=str, help="Dataset location.",
                    default=default_dloc)

def main():
    args = parser.parse_args()

    print("Reading data...")
    set_dataloc(args.dloc)
    metadata = get_metadata()
    graph = GraphContainer(find_meta(args.dataset), args.dloc)

    print("Creating gt.Graph...")
    gt_graph = graph.get_gt_graph()

    assert args.motif_size == 4 or args.motif_size == 3  # Only motif 3 and 4

    all_motif = None
    if args.motif_size == 3:
        if gt_graph.is_directed():
            all_motif = all_3
        else:
            all_motif = all_u3
    else:
        if gt_graph.is_directed():
            all_motif = all_4
        else:
            all_motif = all_u4

    motif_func = None
    if args.num_shuffles <= 0:  # Motif count
        motif_func = motifs
    else:
        motif_func = motif_significance

    output = args.output + str(args.num_shuffles)


    print("Writing scores to file...")
    with open(output, "w") as ofile:
        info = "Dataset: {d} - Motif size: {m} - Directed: {di}\n".format(
                    d=args.dataset, m=args.motif_size,
                    di=str(gt_graph.is_directed()))
        ofile.write(info)

        for i, mc in enumerate(all_motif):
            idx = gt_graph.vertex_index.copy("int")
            shuffle(idx.a)
            g = Graph(gt_graph, vorder=idx)
            if args.num_shuffles <= 0:
                score = motifs(g, k=args.motif_size,
                               motif_list=[mc.gt_motif])[1][0]
            else:
                score = motif_significance(g, k=args.motif_size,
                                           n_shuffles=args.num_shuffles,
                                           motif_list=[mc.gt_motif])[1][0]
            r = "Motif index {}: {}\n".format(i, score)
            print(r)
            ofile.write(r)

    print("Motif analysis for {} is completed.".format(args.dataset))

if __name__ == "__main__":
    main()
