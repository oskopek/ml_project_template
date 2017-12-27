# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A neural network classifier base."""

import argparse
import functools
import os
import sys
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe


class BaseModel(tfe.Network):
    """Base Network."""
    def __init__(self, name=''):
        super(BaseModel, self).__init__(name=name)

    def model_loss(self, labels, images):
        predictions = self.call(images, training=True)

        predictions = tf.argmax(predictions, axis=1, output_type=tf.int64)
        labels = tf.argmax(labels, axis=1, output_type=tf.int64)

        loss_value = self.loss(predictions, labels)
        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', tf.contrib.metrics.accuracy(predictions, labels))
        tf.contrib.summary.scalar('precision', tf.contrib.metrics.precision(predictions, labels))
        tf.contrib.summary.scalar('recall', tf.contrib.metrics.recall(predictions, labels))
        return loss_value

    def loss(self, predictions, labels):
        return tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=labels)

    def train_one_epoch(self, optimizer, dataset, log_interval=None):
        """Trains model on `dataset` using `optimizer`."""

        tf.train.get_or_create_global_step()

        for batch, (images, labels) in enumerate(tfe.Iterator(dataset)):
            with tf.contrib.summary.record_summaries_every_n_global_steps(10):
                batch_model_loss = functools.partial(self.model_loss, labels, images)
                optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
                if log_interval and batch % log_interval == 0:
                    print('Batch #%d\tLoss: %.6f' % (batch, batch_model_loss()))

    def test(self, dataset):
        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tfe.metrics.Mean('loss')
        accuracy = tfe.metrics.Accuracy('accuracy')

        for images, labels in tfe.Iterator(dataset):
            predictions = self.call(images, training=False)
            predictions = tf.argmax(predictions, axis=1, output_type=tf.int64)
            labels = tf.argmax(labels, axis=1, output_type=tf.int64)

            avg_loss(self.loss(predictions, labels))
            accuracy = tf.contrib.metrics.accuracy(predictions, labels)
        print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
              (avg_loss.result(), 100 * accuracy.result()))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', avg_loss.result())
            tf.contrib.summary.scalar('accuracy', accuracy.result())
