import os
import argparse
import numpy as np

from motifwalk.utils import find_meta, set_dataloc, get_metadata, timer
from motifwalk.utils.Graph import GraphContainer
from motifwalk.walks import undirected_randomwalk
from motifwalk.models.skipgram import Skipgram, ADAM



# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str,
                    help="Dataset name (e.g. cora).")
parser.add_argument("-w", "--walk_type", type=str,
                    help="Type of the walk on graph (e.g. random).",
                    default="undirected")
parser.add_argument("-l", "--walk_length", type=int,
                    help="Length of random walk starting from each node.",
                    default=80)
parser.add_argument("-k", "--emb_dim", type=int,
                    help="Embedding dimensionality.", default=16)
parser.add_argument("-t", "--num_step", type=int,
                    help="Number of step to train the embedding.",
                    default=10000)
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
                    help="The initial learning rate.", default=0.001)
parser.add_argument("--log_step", type=int,
                    help="Number of step to report average loss.", default=2000)
parser.add_argument("--save_step", type=int,
                    help="Number of step to save the model.", default=8000)
parser.add_argument("--device", type=str,
                    help="Select device to run the model on using TF format",
                    default="/cpu:0")

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
    if "skipgram" == args.model.lower():
        model = Skipgram(window_size=args.window_size, num_skip=args.num_skip,
                         num_nsamp=args.num_neg)
    elif "gcn" == args.model.lower():
        print ("TODO")
    elif "sc" == args.model.lower():
        print ("TODO")
    else:
        print("Unknown embedding model.")
    assert model is not None
    model.build(num_vertices=gt.num_vertices(), emb_dim=args.emb_dim,
                batch_size=args.batch_size, learning_rate=args.learning_rate)
    timer(False)

    print("Generating walks...")
    timer()
    walks = None
    index = None
    if "undirected" == args.walk_type:
        walks, index = undirected_randomwalk(gt)
    else:
        print("TODO")
    assert walks is not None
    timer(False)

    print("Start training...")
    timer()
    emb = model.train(data=walks, num_step=args.num_step,
                      log_step=args.log_step, save_step=args.save_step,
                      learning_rate=args.learning_rate)
    timer(False)

    from time import time
    uid = str(time())
    np.save("{}_{}.emb.py".format(args.dataset, uid), emb)
    with open("{}_{}.info".format(args.dataset, uid), "w") as infofile:
        infofile.write(uid + '\n')
        args_dict = vars(args)
        for key, val in args_dict.items():
            infofile.write("{}: {}\n".format(key,val))

if __name__ == "__main__":
    main()
