import tensorflow as tf
import math
import collections
from motifwalk.models import EmbeddingModel
from tensorflow import train
import numpy as np
from numpy.random import randint, seed

seed(42)

GDO = train.GradientDescentOptimizer
ADAM = train.AdamOptimizer

class Skipgram(EmbeddingModel):

    def __init__(self, window_size, num_skip, num_nsamp):
        """Initialize a Skipgram embedding model. Examples of this
        class is DeepWalk and node2vec.

        Parameters:
        window_size - int - Size of the skip window
        num_skip - int - Number of samples generated per window
        num_nsamp - int - Number of negative samples per window
        batch_size - int - Size of each batch training batch
        """
        super().__init__()
        self.window_size = window_size
        self.num_skip = num_skip
        self.num_nsamp = num_nsamp
        self.device = '/cpu:0'
        self.tf_graph = None
        self.embedding = None
        self.init_op = None
        self.train_inputs = None
        self.train_labels = None
        self.optimizer = None
        self.loss = None
        self.batch_size = None
        self.data_index = 0 # Pointer to data

    def build(self, num_vertices, emb_dim=16, batch_size=1024,
              opt=GDO, learning_rate=0.01, force_rebuild=False):
        """Build the computing graph.

        Parameters:
        num_vertices - int - Number of vertices in the target network
        emb_dim - int - The dimensionality of the embedding vectors
        batch_size - int - Mini batch training size
        force_rebuild - bool - Force rebuild an existing model

        Returns:
        init_op - tf.Ops - Operation to intialize all variables
        embed - tf.Variable - Set of embedding vectors in a batch
        nce_weights - tf.Variable - Set of context embedding vectors
        nce_biases - tf.Variable - Set of biases (normilization factors)
        """
        if self.tf_graph is not None and not force_rebuild:
            print("The computation graph is already built.")
            return self.tf_graph
        graph = tf.Graph()
        with graph.as_default():
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
            with tf.device(self.device):
                embeddings = tf.Variable(
                                tf.random_uniform([num_vertices, emb_dim],
                                                  -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                nce_weights = tf.Variable(
                                 tf.truncated_normal([num_vertices, emb_dim],
                                                 stddev=1.0/math.sqrt(emb_dim)))
                nce_biases = tf.Variable(tf.zeros([num_vertices]))
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1,
                                             keep_dims=True))
                normalized_embeddings = embeddings / norm
                nce_loss = self._loss(embed, nce_weights, nce_biases,
                                      train_labels, num_vertices)
                optimizer = opt(learning_rate).minimize(nce_loss)
                init_op = tf.global_variables_initializer()
                self.tf_graph = graph
                self.init_op = init_op
                self.train_inputs = train_inputs
                self.train_labels = train_labels
                self.optimizer = optimizer
                self.embedding = normalized_embeddings
                self.loss = nce_loss
                self.batch_size = batch_size

    def _loss(self, embed, nce_weights, nce_biases, train_labels, num_vertices):
        l = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                               labels=train_labels, inputs=embed,
                               num_sampled=self.num_nsamp,
                               num_classes=num_vertices) )
        return l


    def train(self, data, num_step, log_step, save_step,
              opt=GDO, learning_rate=None, retrain=False):
        """Train the model. TODO: Implement session recovering.
        """
        if self.tf_graph is None:
            print("Build graph first!")
            return None
        with tf.Session(graph=self.tf_graph) as session:
            # Update learning_rate if needed
            if learning_rate is not None:
                self.optimizer = opt(learning_rate).minimize(self.loss)
            # Skip variables initialization if fine tuning models
            if not retrain:
                session.run(self.init_op)
                print("All variables of Skipgram model is initialized.")
            average_loss = 0
            for step in range(num_step):
                batch_inputs, batch_labels = self.generate_batch(data)
                feed_dict = {self.train_inputs: batch_inputs,
                             self.train_labels: batch_labels}
                _, loss_val = session.run([self.optimizer, self.loss],
                                          feed_dict=feed_dict)
                average_loss += loss_val
                if step % log_step == 0:
                    if step > 0:
                        average_loss /= log_step
                    print("Average loss at step {}: {}".format(
                                                step, average_loss))
                    average_loss = 0
                if step % save_step == 0:
                    if step > 0:
                        pass
                    # TODO: Save embedding every step
            return self.embedding.eval()

    def generate_batch(self, data):
        """Generate data for training."""
        batch_size = self.batch_size
        num_skip = self.num_skip
        window_size = self.window_size
        assert batch_size % num_skip == 0
        assert num_skip <= 2 * window_size
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * window_size + 1
        buf = collections.deque(maxlen=span)
        for _ in range(span):
            buf.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % data.size
        for i in range(batch_size // num_skip):
            target = window_size
            targets_to_avoid = [window_size]
            for j in range(num_skip):
                while target in targets_to_avoid:
                    target = randint(0, span-1)
                targets_to_avoid.append(target)
                batch[i * num_skip + j] = buf[window_size]
                labels[i * num_skip + j, 0] = buf[target]
                self.data_index = (self.data_index + 1) % data.size
        return batch,labels
