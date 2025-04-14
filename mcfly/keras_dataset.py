import keras
import math
import numpy as np


class NumpyKerasDataset(keras.utils.PyDataset):
    # References:
    # - https://keras.io/api/utils/python_utils/#pydataset-class
    # - https://stackoverflow.com/a/70319612

    def __init__(
        self,
        x_set: np.ndarray,
        y_set: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # initialization
        self.batch_size = batch_size
        self.x = x_set
        self.y = y_set
        self.datalen = len(x_set)
        self.indexes = np.arange(self.datalen)
        if shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def __len__(self):
        # denotes the number of batches per epoch
        return math.ceil(self.datalen / self.batch_size)
