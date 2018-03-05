'''
Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

Adapted from: https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
'''

from models.base import KerasModel

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Flags
from flags import flags_parser
FLAGS = flags_parser.FLAGS
assert FLAGS is not None


# Model
class MnistCnn(KerasModel):
    # Setup constants
    IMAGE_SIZE = 28

    def __init__(self):
        super(MnistCnn, self).__init__(
            logdir=FLAGS.data.out_dir, expname="MNIST-CNN", threads=FLAGS.training.threads, seed=FLAGS.training.seed)

    # Construct the graph
    def _build(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(FLAGS.model.arch.num_classes, activation='softmax'))

        optimizer = keras.optimizers.Adadelta(lr=FLAGS.model.optimization.learning_rate)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        self.model = model

    def _load_data(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.IMAGE_SIZE, self.IMAGE_SIZE)
            x_test = x_test.reshape(x_test.shape[0], 1, self.IMAGE_SIZE, self.IMAGE_SIZE)
            self.input_shape = (1, self.IMAGE_SIZE, self.IMAGE_SIZE)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
            x_test = x_test.reshape(x_test.shape[0], self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
            self.input_shape = (self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, FLAGS.model.arch.num_classes)
        y_test = keras.utils.to_categorical(y_test, FLAGS.model.arch.num_classes)
        return (x_train, y_train), (x_test, y_test)

    def run(self):
        (x_train, y_train), (x_test, y_test) = self._load_data()
        self._build()

        self.model.fit(
            x_train,
            y_train,
            batch_size=FLAGS.model.optimization.batch_size,
            epochs=FLAGS.model.optimization.epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=self.callbacks)

        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


# Runner method
def run():
    MnistCnn().run()
