import tensorflow as tf
import os

global FLAGS
FLAGS = tf.app.flags.FLAGS


def define_flags():
    # Directories
    tf.app.flags.DEFINE_string('in_data_dir', 'data_in', "Directory from which to read input datasets.")
    tf.app.flags.DEFINE_string(
        'out_data_dir', 'data_out', """Directory where to write event logs """
        """and checkpoint."""
    )
    if os.name == 'nt':
        tf.app.flags.DEFINE_string(
            'checkpoint_dir', 'e:/temp/tensorflow/checkpoints/', 'Directory to save checkpoints in (once per epoch)'
        )
    else:
        chkpt_dir = '/tmp/tensorflow/checkpoints/'
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        tf.app.flags.DEFINE_string('checkpoint_dir', chkpt_dir, 'Directory to save checkpoints in (once per epoch)')

    # TF parameters
    tf.app.flags.DEFINE_boolean("no_gpu", False, 'Disables GPU usage even if a GPU is available')
    tf.app.flags.DEFINE_integer("log_interval", 100, 'How many batches to wait before logging training statistics.')

    # Optimization parameters
    tf.app.flags.DEFINE_integer('epochs', 50, 'Training epoch count')
    tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
    tf.app.flags.DEFINE_float('momentum', 0.5, 'SGD momentum')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'Training batch size')

    # Model parameters
    tf.app.flags.DEFINE_integer('conv_size', 32, 'First convolution layer size')

    # Jupyter notebook params
    # Only to avoid raising UnrecognizedFlagError
    tf.app.flags.DEFINE_string('f', 'kernel', 'Kernel')
