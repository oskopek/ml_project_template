import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

from datetime import datetime


# Make sure to implement the run() function in the model file!
class BaseModel:
    def __init__(self, logdir="logs", expname="exp", threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads)
        )

        # Construct the graph
        with self.session.graph.as_default():
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.is_training = tf.placeholder_with_default(False, [])

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            logdir = "{}/{}-{}".format(logdir, expname, timestamp)
            self.summary_writer = tf_summary.create_file_writer(logdir, flush_millis=5 * 1000)

    def _init_variables(self):
        # Initialize variables
        self.session.run(tf.global_variables_initializer())
        with self.summary_writer.as_default():
            tf_summary.initialize(session=self.session, graph=self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)
