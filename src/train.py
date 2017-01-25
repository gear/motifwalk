"""coding=utf-8
"""

import tensorflow as tf
import collections
import pickle as p
import numpy as np
import random
from time import time
import re
from os import path
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from walks import WalkGenerator
from constrains import R, UTriangle, UWedge


context_ext = ".{}_context"
walk_type_ = {'random': R, 'triangle': UTriangle, 'wedge': UWedge}
data_index = 0


def prob_type(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def generate_graph_context(gfile, args):
    """Generate and return a graph context data file.
    Parameters:
    ===========
        gfile: graph pickle file name
        walk_type: type of random walk
        ef: walk bias enforcement
    Returns:
    ========
        cfile: graph context file name"""
    walk_type = args.walk_type
    num_walk = args.num_walk
    walk_length = args.walk_length
    ef = args.walk_bias
    with open(gfile+".data", 'rb') as pf:
        graph_data = p.load(pf)    
    graph = graph_data['NXGraph']
    if not args.directed:
        if graph.is_directed():
            graph = graph.to_undirected()
    pattern = walk_type_[walk_type]
    walker = WalkGenerator(graph=graph, constrain=pattern(enforce_prob=ef))
    with open(gfile + context_ext.format(walk_type), 'w') as f:
        for i in walker(walk_length=walk_length, 
                        num_walk=num_walk, yield_size=80):
            f.write(' '.join(map(str, i)) + '\n')
    print("Wrote graph context file {} to\
           disk.".format(gfile + context_ext.format(walk_type)))
    return gfile + context_ext.format(walk_type)


def build_dataset(context, graph_size):
    """Read data and build dataset for skipgram training."""
    count = []
    count.extend(collections.Counter(context).most_common(graph_size))
    data = list()
    for node in context:
        data.append(int(node))
    return data, count


def generate_batch(data, args):
    """Generate training batches for skipgram model."""
    batch_size = args.batch_size
    num_skip = args.num_skip
    window_size = args.window_size
    global data_index
    assert batch_size % num_skip == 0
    assert num_skip <= 2 * window_size
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * window_size + 1
    buf = collections.deque(maxlen=span)
    for _ in range(span):
        buf.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skip):
        target = window_size
        targets_to_avoid = [window_size]
        for j in range(num_skip):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i * num_skip + j] = buf[window_size]
            labels[i * num_skip + j, 0] = buf[target] 
            data_index = (data_index + 1) % len(data)
    return batch,labels


def run_embeddings(data, args): 
    """Construct tensorflow computation graph and train the model."""
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[args.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[args.batch_size,1])
        with tf.device('/cpu:0'):
            embeddings = tf.get_variable("emb", shape=[args.graph_size, args.emb_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weights = tf.get_variable("nce", shape=[args.graph_size, args.emb_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
            nce_biases = tf.Variable(tf.zeros([args.graph_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                             labels=train_labels, inputs=embed,
                                             num_sampled=args.num_nsamp,
                                             num_classes=args.graph_size))
        optimizer = tf.train.AdadeltaOptimizer(args.learning_rate).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        init = tf.initialize_all_variables()
    with tf.Session(graph=graph) as session:
        init.run()
        print("Initialized")
        average_loss = 0
        for step in range(args.num_step):
            batch_inputs, batch_labels = generate_batch(data, args)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings

     
def main():
    """Parse arguments and run the embedding algorithms"""
    parser = ArgumentParser("mage",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--input", nargs='?', required=True,
                        help="Input graph file name")
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
    parser.add_argument("--directed", default=False, type=bool,
                        help="Is the graph directed")
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
    parser.add_argument("--graph-size", default=10312, type=int,
                        help="Number of nodes for embedding")
    parser.add_argument("--num-step", default=6000000, type=int,
                        help="Number of batches used for training")
    parser.add_argument("--learning-rate", default=0.1, type=float,
                        help="Adagrad learning rate")
    args = parser.parse_args()

    print("Checking if the graph context already exists...")
    graph_name = re.match(r"(.+)\.(.+)", args.input).group(1)
    context_file = (graph_name + context_ext).format(args.walk_type)
    if path.exists(context_file):
        print("Found graph context file: {}.\
               Skipping context generation...".format(context_file))
        context_file = open(context_file)
    else:
        print("Walking... This operation might take a while...")
        context_file = open(generate_graph_context(graph_name, args))
    context = tf.compat.as_str(context_file.read()).split()
    data, _ = build_dataset(context, args.graph_size) 
    del context  # Save memory
    t0 = time()
    embed = run_embeddings(data, args)
    print("Finished training in {}.".format(time()-t0))
    print("Writing result to file...")
    with open(args.output, 'w') as f:
        f.write("{} {}\n".format(*embed.shape))
        for idx, i in enumerate(embed):
            f.write(str(idx) + ' ' + ' '.join(str(j) for j in i) + '\n')
    print("Done!")

if __name__ == "__main__":
    main()
