import numpy as np


class DataBatch(object):
    def __init__(self, x, y, shuffle=False):
        assert len(x) == len(y)

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        self._size = len(x)
        if shuffle:
            index = np.random.permutation(self._size)
            x = x[index]
            y = y[index]
        self._X = x
        self._y = y
        self._shuffle = shuffle
        self._index = 0

    def next_batch(self, batch_size):
        start = self._index
        self._index += batch_size
        if self._index > self._size:
            start = 0
            if self._shuffle:
                index = np.random.permutation(self._size)
                self._X = self._X[index]
                self._y = self._y[index]
            self._index = start + batch_size
            assert batch_size <= self._size
        end = self._index
        return self._X[start: end], self._y[start: end]

    @property
    def x(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def size(self):
        return self._size