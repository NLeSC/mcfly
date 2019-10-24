import os
import unittest

import numpy as np
from tensorflow.keras.utils import to_categorical

from mcfly import find_architecture, storage
from test_tools import save_remove


class StorageSuite(unittest.TestCase):
    """Basic test cases."""

    def test_savemodel(self):
        """ Test whether a dummy model is saved """
        model = create_dummy_model()

        storage.savemodel(model, self.path, self.modelname)

        assert os.path.isfile(self.architecture_json_file_name) and os.path.isfile(self.weights_file_name)

    def test_savemodel_keras(self):
        """ Test whether a dummy model is saved """
        model = create_dummy_model()

        model.save(self.keras_model_file_path)

        assert os.path.isfile(self.keras_model_file_path)

    def test_loadmodel(self):
        """ Test whether a dummy model can be save and then loaded """
        model = create_dummy_model()
        storage.savemodel(model, self.path, self.modelname)

        loaded_model = storage.loadmodel(self.path, self.modelname)

        assert hasattr(loaded_model, 'fit')

    def setUp(self):
        self.path = os.getcwd() + '/'
        self.modelname = 'teststorage'
        self.architecture_json_file_name = self.path + self.modelname + '_architecture.json'
        self.weights_file_name = self.path + self.modelname + '_weights.npy'
        self.keras_model_file_path = os.path.join(self.path, 'teststorage.h5')

    def tearDown(self):
        save_remove(self.architecture_json_file_name)
        save_remove(self.weights_file_name)
        save_remove(self.keras_model_file_path)


def create_dummy_model():
    np.random.seed(123)
    num_timesteps = 100
    num_channels = 2
    num_samples_train = 5
    num_samples_val = 3
    X_train = np.random.rand(
        num_samples_train,
        num_timesteps,
        num_channels)
    y_train = to_categorical(np.array([0, 0, 1, 1, 1]))
    X_val = np.random.rand(num_samples_val, num_timesteps, num_channels)
    y_val = to_categorical(np.array([0, 1, 1]))
    best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
        X_train, y_train, X_val, y_val, verbose=False, subset_size=10,
        number_of_models=1, nr_epochs=1)
    return best_model


if __name__ == '__main__':
    unittest.main()
