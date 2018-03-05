import numpy as np
import tensorflow as tf


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
