import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary as tf_summary

from models.base import BaseModel

# Flags
from flags import flags_parser
FLAGS = flags_parser.FLAGS
assert FLAGS is not None


# Utils
def next_batch(arr, batch_size):
    num_batches = int(len(arr) / batch_size)
    for i in range(0, num_batches * batch_size, batch_size):
        yield arr[i:i + batch_size]
    yield arr[num_batches * batch_size:]


def noise(size, dist='uniform'):
    if dist == 'uniform':
        return np.random.uniform(-1, 1, size=size)
    elif dist == 'normal':
        return np.random.normal(size=size)
    elif dist == 'linspace':
        n, dim = np.sqrt(size[0]).astype(np.int32), size[1]
        interpolated_noise = []
        starts, ends = noise((n, dim)), noise((n, dim))
        for i in range(n):
            for w in np.linspace(0, 1, n):
                interpolated_noise.append(starts[i] + (ends[i] + starts[i]) * w)
        return np.asarray(interpolated_noise)


def shuffle(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    return a[permutation], b[permutation]


def tile_images(images, num_x, num_y, h, w):
    res = tf.zeros((num_y * h, num_x * w))
    index = -1
    rows = []
    for i in range(0, num_y):
        row = []
        for j in range(0, num_x):
            index += 1
            row.append(tf.reshape(images[index], (h, w)))
        rows.append(tf.concat(row, 1))
    res = tf.concat(rows, 0)
    print("res shape:", res.shape)
    return tf.reshape(res, (1, num_y * h, num_x * w, 1))


# Discriminator
def discriminator(X, reuse):
    with tf.variable_scope("Discriminator", reuse=reuse):
        # Layer 1
        dx = tf.layers.dense(
            X,
            units=1024,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            activation=tf.nn.relu,
            name='fc1'
        )
        # Layer 2
        # dx = tf.layers.dense(dx, units=512, activation=tf.nn.relu, name='fc2')
        # Layer 3
        # dx = tf.layers.dense(dx, units=256, activation=tf.nn.relu, name='fc3')
        # Layer 4
        d_out = tf.layers.dense(
            dx, units=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc_out'
        )
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


# Model
class MnistGan(BaseModel):
    # Setup constants
    IMAGE_PIXELS = 784
    NOISE_SIZE = 100

    def __init__(self):
        super(MnistGan, self).__init__(
            logdir=FLAGS.data.out_dir, expname="MNIST-GAN", threads=FLAGS.training.threads, seed=FLAGS.training.seed
        )
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

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))
        )
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake))
        )
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake))
        )

        # Test summaries
        tiled_image_random = tile_images(g_sample, 6, 6, 28, 28)
        tiled_image_interpolated = tile_images(generator(self.noise_input_interpolated, reuse=True), 6, 6, 28, 28)
        with self.summary_writer.as_default(), tf_summary.always_record_summaries():
            gen_image_summary_op = tf_summary.image(
                'generated_images', tiled_image_random, max_images=1, step=self.g_step
            )
            gen_image_summary_interpolated_op = tf_summary.image(
                'generated_images_interpolated', tiled_image_interpolated, max_images=1, step=self.g_step
            )
            self.IMAGE_SUMMARIES = [gen_image_summary_op, gen_image_summary_interpolated_op]

        # Optimizers
        t_vars = tf.trainable_variables()
        self.d_opt = tf.train.AdamOptimizer(2e-4).minimize(
            self.d_loss, var_list=[var for var in t_vars if 'Discriminator' in var.name], global_step=self.d_step
        )
        self.g_opt = tf.train.AdamOptimizer(2e-4).minimize(
            self.g_loss, var_list=[var for var in t_vars if 'Generator' in var.name], global_step=self.g_step
        )

        # saver = tf.train.Saver(max_to_keep=1)

    def load_data(self):
        # Read the input data
        # In this case, MNIST + batch and shuffle it. In our case, it will be quite different.

        from tensorflow.examples.tutorials.mnist import input_data

        def read_data_sets(data_dir):
            """Returns training and test tf.data.Dataset objects."""
            data = input_data.read_data_sets(data_dir, one_hot=True)
            # train_ds = tf.data.Dataset.from_tensor_slices((data.train.images,
            #                                               data.train.labels))
            # test_ds = tf.data.Dataset.from_tensors(
            #   (data.test.images, data.test.labels))
            return (data.train, data.test)

        device, data_format = ('/gpu:0', 'channels_first')
        if FLAGS.training.no_gpu:
            device, data_format = ('/cpu:0', 'channels_last')
        print('Using device %s, and data format %s.' % (device, data_format))

        # Load the datasets
        train_ds, test_ds = read_data_sets(FLAGS.data.in_dir)
        # train_ds = train_ds.shuffle(60000).batch(FLAGS.batch_size)
        return shuffle(train_ds.images, train_ds.labels)

    def train_batch(self, batch):
        BATCH_SIZE = FLAGS.model.optimization.batch_size

        # 1. Train Discriminator
        # X_batch = images_to_vectors(batch.permute(0, 2, 3, 1).numpy())
        batch_noise = noise((BATCH_SIZE, self.NOISE_SIZE))
        feed_dict = {self.images_input: batch, self.noise_input: batch_noise}
        d_error, _ = self.session.run([self.d_loss, self.d_opt], feed_dict=feed_dict)

        # 2. Train Generator
        feed_dict = {self.noise_input: batch_noise}
        g_error, _ = self.session.run([self.g_loss, self.g_opt], feed_dict=feed_dict)

        return d_error, g_error

    # Generate images from test noise
    def test_eval(self, epoch_info, test_noise_random, test_noise_interpolated):
        self.session.run(
            self.IMAGE_SUMMARIES,
            feed_dict={
                self.noise_input: test_noise_random,
                self.noise_input_interpolated: test_noise_interpolated
            }
        )

    def run(self):
        BATCH_SIZE = FLAGS.model.optimization.batch_size
        train_X, train_Y = self.load_data()

        num_test_samples = 36
        test_noise_random = noise((num_test_samples, self.NOISE_SIZE))
        test_noise_interpolated = noise((num_test_samples, self.NOISE_SIZE), dist='linspace')

        # Iterate through epochs
        for epoch in range(FLAGS.model.optimization.epochs):
            print("Epoch %d" % epoch)
            for n_batch, batch in enumerate(next_batch(train_X, BATCH_SIZE)):
                d_error, g_error = self.train_batch(batch)

                # Test noise
                if n_batch % 500 == 0:
                    # display.clear_output(True)
                    self.test_eval((epoch, n_batch), test_noise_random, test_noise_interpolated)
                    print("Epoch: {}, Batch: {}, D_Loss: {}, G_Loss: {}".format(epoch, n_batch, d_error, g_error))


# Runner method
def run():
    MnistGan().run()
