import math
import numpy

from keras.datasets import imdb
from keras import utils


WORDS = 1000
LENGTH = 200


class Sequence(utils.Sequence):

    def __init__(self, X, y, batch_size, indices=None, test=False):
        self.X = X
        self.y = y.astype('f')
        self.batch_size = batch_size
        self.indices = indices or list(range(len(X)))
        self.test = test

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = begin + self.batch_size
        batch_idx = self.indices[begin: end]
        batch_x = numpy.array([(x + [0] * LENGTH)[:LENGTH] for x in self.X[batch_idx]])
        batch_y = self.y[batch_idx]
        return batch_x, batch_y


def batch_generator(batch_size, validation_split=0.1, test=False):

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=WORDS)
    print((x_train.shape, y_train.shape), (x_test.shape, y_test.shape))

    if not test:
        num = len(x_train)
        indices = list(range(num))
        num_valid = int(num * validation_split)
        indices_train = indices[num_valid:]
        indices_valid = indices[:num_valid]
        seq_train = Sequence(x_train, y_train, batch_size, indices=indices_train)
        seq_valid = Sequence(x_train, y_train, batch_size, indices=indices_valid)
        return seq_train, seq_valid

    else:
        return Sequence(x_test, y_test, batch_size, test=True)
