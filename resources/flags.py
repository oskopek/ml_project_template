import tensorflow as tf
import os
import sys

global FLAGS
FLAGS = tf.app.flags.FLAGS


def define_flags():
    # Directories
    tf.app.flags.DEFINE_string('in_data_dir', 'data_in',
                        "Directory from which to read input datasets.")
    tf.app.flags.DEFINE_string('out_data_dir', 'data_out',
                           """Directory where to write event logs """
                           """and checkpoint.""")
    if os.name == 'nt':
        tf.app.flags.DEFINE_string('checkpoint_dir', 'e:/temp/tensorflow/checkpoints/',
                        'Directory to save checkpoints in (once per epoch)')
    else:
        tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/tensorflow/checkpoints/',
                        'Directory to save checkpoints in (once per epoch)')    

    # TF parameters
    tf.app.flags.DEFINE_boolean("no_gpu", False, 'Disables GPU usage even if a GPU is available')
    tf.app.flags.DEFINE_integer("log_interval", 100, 'How many batches to wait before logging training statistics.')

    # Optimization parameters
    tf.app.flags.DEFINE_integer('epochs', 10, 'Training epoch count')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
    tf.app.flags.DEFINE_float('momentum', 0.5, 'SGD momentum')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'Training batch size')

    # Model parameters
    tf.app.flags.DEFINE_integer('conv_size', 32, 'First convolution layer size')
    
    # Jupyter notebook params
    tf.app.flags.DEFINE_string('f', 'kernel', 'Kernel')  # Only to avoid raising UnrecognizedFlagError
