import tensorflow as tf

DIS_VAR_PREFIX = "Discriminator"
GEN_VAR_PREFIX = 'Generator'


def discriminator_simple(X, reuse):
    """ The simplest discriminator. """
    with tf.variable_scope(DIS_VAR_PREFIX, reuse=reuse):
        # Layer 1
        dx = tf.layers.dense(
            X,
            units=1024,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            activation=tf.nn.relu,
            name='fc1')
        # Last layer
        d_out = tf.layers.dense(
            dx, units=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='fc_out')
        return d_out


def generator_simple(X, reuse=False):
    """ The simplest discriminator. """
    with tf.variable_scope(GEN_VAR_PREFIX, reuse=reuse):
        # Layer 1
        gx = tf.layers.dense(X, units=128, activation=tf.nn.relu, name='fc1')
        # Last layer
        g_out = tf.layers.dense(gx, units=784, activation=tf.nn.sigmoid, name='fc_out')
        return g_out
