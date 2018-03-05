import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

from models.base import BaseModel
from resources.data_utils import next_batch, read_mnist
from resource.model_utils import noise, tile_images

# Flags
from flags import flags_parser
FLAGS = flags_parser.FLAGS
assert FLAGS is not None


# Discriminator
def discriminator(X, reuse):
    with tf.variable_scope("Discriminator", reuse=reuse):
        # Layer 1
        dx = tf.layers.dense(
            X,
            units=1024,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            activation=tf.nn.relu,
            name='fc1')
        # Layer 2
        # dx = tf.layers.dense(dx, units=512, activation=tf.nn.relu, name='fc2')
        # Layer 3
        # dx = tf.layers.dense(dx, units=256, activation=tf.nn.relu, name='fc3')
        # Layer 4
        d_out = tf.layers.dense(
            dx, units=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc_out')
        return d_out


# Generator
def generator(X, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # Layer 1
        gx = tf.layers.dense(X, units=128, activation=tf.nn.relu, name='fc1')
        # Layer 2
        # gx = tf.layers.dense(gx, units=512, activation=tf.nn.relu, name='fc2')
        # Layer 3
        # gx = tf.layers.dense(gx, units=1024, activation=tf.nn.relu, name='fc3')
        # Layer 4
        g_out = tf.layers.dense(gx, units=784, activation=tf.nn.sigmoid, name='fc_out')
        return g_out


# Shortcut for cross-entropy loss calculation.
def cross_entropy_loss(logits=None, labels=None):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


# Model
class MnistGan(BaseModel):
    # Setup constants
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    NOISE_SIZE = 100

    def __init__(self):
        super(MnistGan, self).__init__(
            logdir=FLAGS.data.out_dir, expname="MNIST-GAN", threads=FLAGS.training.threads, seed=FLAGS.training.seed)
        with self.session.graph.as_default():
            self._build()
            self._init_variables()

    # Construct the graph
    def _build(self):
        self.d_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="d_step")
        self.g_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="g_step")

        self.images_input = tf.placeholder(tf.float32, shape=(None, self.IMAGE_PIXELS))
        self.noise_input = tf.placeholder(tf.float32, shape=(None, self.NOISE_SIZE))
        self.noise_input_interpolated = tf.placeholder(tf.float32, shape=(None, self.NOISE_SIZE))

        # Losses
        g_sample = generator(self.noise_input)
        d_real = discriminator(self.images_input, reuse=False)
        d_fake = discriminator(g_sample, reuse=True)

        d_loss_real = cross_entropy_loss(logits=d_real, labels=tf.ones_like(d_real))
        d_loss_fake = cross_entropy_loss(logits=d_fake, labels=tf.zeros_like(d_fake))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = cross_entropy_loss(logits=d_fake, labels=tf.ones_like(d_fake))

        # Test summaries
        tiled_image_random = tile_images(g_sample, 6, 6, self.IMAGE_SIZE, self.IMAGE_SIZE)
        tiled_image_interpolated = tile_images(
            generator(self.noise_input_interpolated, reuse=True), 6, 6, self.IMAGE_SIZE, self.IMAGE_SIZE)
        with self.summary_writer.as_default(), tf_summary.always_record_summaries():
            gen_image_summary_op = tf_summary.image(
                'generated_images', tiled_image_random, max_images=1, step=self.g_step)
            gen_image_summary_interpolated_op = tf_summary.image(
                'generated_images_interpolated', tiled_image_interpolated, max_images=1, step=self.g_step)
            self.IMAGE_SUMMARIES = [gen_image_summary_op, gen_image_summary_interpolated_op]

        # Optimizers
        t_vars = tf.trainable_variables()
        LEARNING_RATE = FLAGS.training.model.optimization.learning_rate
        self.d_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
            self.d_loss, var_list=[var for var in t_vars if 'Discriminator' in var.name], global_step=self.d_step)
        self.g_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
            self.g_loss, var_list=[var for var in t_vars if 'Generator' in var.name], global_step=self.g_step)

        # saver = tf.train.Saver(max_to_keep=1) # TODO(jendelel): Set up saver.

    def train_batch(self, batch):
        BATCH_SIZE = FLAGS.model.optimization.batch_size

        # 1. Train Discriminator
        batch_noise = noise((BATCH_SIZE, self.NOISE_SIZE))
        feed_dict = {self.images_input: batch, self.noise_input: batch_noise}
        d_error, _ = self.session.run([self.d_loss, self.d_opt], feed_dict=feed_dict)

        # 2. Train Generator
        feed_dict = {self.noise_input: batch_noise}
        g_error, _ = self.session.run([self.g_loss, self.g_opt], feed_dict=feed_dict)

        return d_error, g_error

    # Generate images from test noise
    def test_eval(self, noise_input, noise_input_interpolated):
        self.session.run(
            self.IMAGE_SUMMARIES,
            feed_dict={
                self.noise_input: noise_input,
                self.noise_input_interpolated: noise_input_interpolated
            })

    def run(self):
        BATCH_SIZE = FLAGS.model.optimization.batch_size
        train_X, train_Y = read_mnist(FLAGS.data.in_dir, no_gpu=FLAGS.training.no_gpu)

        test_noise_random = noise(size=(FLAGS.eva.num_test_samples, self.NOISE_SIZE), dist='uniform')
        test_noise_interpolated = noise(size=(FLAGS.eva.num_test_samples, self.NOISE_SIZE), dist='linspace')

        # Iterate through epochs
        for epoch in range(FLAGS.model.optimization.epochs):
            print("Epoch %d" % epoch, flush=True)
            for n_batch, batch in enumerate(next_batch(train_X, BATCH_SIZE)):
                d_error, g_error = self.train_batch(batch)

                # Test noise
                if n_batch % FLAGS.training.log_interval == 0:
                    self.test_eval(test_noise_random, test_noise_interpolated)
                    print(
                        "Epoch: {}, Batch: {}, D_Loss: {}, G_Loss: {}".format(epoch, n_batch, d_error, g_error),
                        flush=True)


# Runner method
def run():
    MnistGan().run()
