import numpy as np


def next_batch(arr, batch_size):
    num_batches = int(len(arr) / batch_size)
    for i in range(0, num_batches * batch_size, batch_size):
        yield arr[i:i + batch_size]
    # Yield the remainder of the batch.
    if num_batches * batch_size != len(arr):
        yield arr[num_batches * batch_size:]


def shuffle(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    return a[permutation], b[permutation]


def read_mnist(in_dir, no_gpu=False):
    # In this case, MNIST + batch and shuffle it. In our case, it will be quite different.

    from tensorflow.examples.tutorials.mnist import input_data

    def read_data_sets(data_dir):
        """Returns training and test tf.data.Dataset objects."""
        data = input_data.read_data_sets(data_dir, one_hot=True)
        # train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels))
        # test_ds = tf.data.Dataset.from_tensors((data.test.images, data.test.labels))
        return (data.train, data.test)

    device, data_format = ('/gpu:0', 'channels_first')
    if no_gpu:
        device, data_format = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    # Load the datasets
    train_ds, test_ds = read_data_sets(in_dir)
    # train_ds = train_ds.shuffle(60000).batch(batch_size)
    return shuffle(train_ds.images, train_ds.labels)
