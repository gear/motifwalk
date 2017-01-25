"""coding=utf-8
"""

import tensorflow as tf
import collections
import pickle as p
from re import match
from os import path
from argparse import ArgumentParser, FileType, 
                     ArgumentDefaultHelpFormatter, ArgumentTypeError
from walks import WalkGenerator
from constrains import R, UTriangle, UWedge


context_ext = ".{}_context"
walk_type_ = {'random': R, 'triangle': UTriangle, 'wedge': UWedge}


def prob_type(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def generate_graph_context(gfile, walk_type, ef=0.9):
    """Generate and return a graph context data file.
    Parameters:
    ===========
        gfile: graph pickle file name
        walk_type: type of random walk
        ef: walk bias enforcement
    Returns:
    ========
        cfile: graph context file name"""
    with open(gfile, 'rb') as pf:
        graph_data = p.load(pf)    
    graph = graph_data['NXGraph']
    pattern = walk_type_[walk_type]
    walker = WalkGenerator(graph=graph, constrain=pattern())
     
def main():
    """Parse arguments and run the embedding algorithms"""
    parser = ArgumentParser("mage",
                            formatter_class=ArgumentDefaultHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--format", default="edgelist",
                        help="Input file format")
    parser.add_argument("--input", nargs='?', required=True,
                        help="Input graph file")
    parser.add_argument("--output", required=True,
                        help="Output embedding file")
    parser.add_argument("--emb_dim", default=128, type=int,
                        help="Embedding dimensionality")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed")
    parser.add_argument("--num-walk", default=10, type=int,
                        help="Number of walks for each node")
    parser.add_argument("--walk-length", default=80, type=int,
                        help="Length of the random walk starting on each node")
    parser.add_argument("--window-size", default=4, type=int,
                        help="Window size for skipgram model")
    parser.add_argument("--num-skip", default=4, type=int,
                        help="Number of generated samples for each skip-window")
    parser.add_argument("--num-nsamp", default=64, type=int,
                        help="Number of negative samples for NCE estimation")
    parser.add_argument("--walk-type", default='triangle', type=str,
                        help="Type of random walk for context generation")
    parser.add_argument("--walk-bias", default=0.9, type=prob_type,
                        help="Level of enforcing the motif walk")
    parser.add_argument("--batch-size", default=128, type=int,
                        help="Batch size for model training")
    args = parser.parse_args()

    print("Checking if the graph context already exists...")
    graph_name = re.match(r"(.+)\.(.+)", args.input).group(1)
    context_file = (graph_name + context_ext).format(args.walk_type)
    if path.exists(context_file):
        print("Found graph context file: {}. Skipping context generation...".format(context_file)) 
        context_file = open(context_file)
    else:
        context_file = open(generate_graph_context(graph_name+".data", 
                                                   args.walk_type, args.walk_bias))
    data = tf.compat.as_str(context_file.read()).split()

