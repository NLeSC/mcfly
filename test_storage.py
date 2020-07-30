import os
import unittest

import numpy as np

from mcfly import storage, modelgen
from test_tools import safe_remove


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
        safe_remove(self.architecture_json_file_name)
        safe_remove(self.weights_file_name)
        safe_remove(self.keras_model_file_path)


def create_dummy_model():
    np.random.seed(123)
    num_time_steps = 100
    num_channels = 2
    num_samples_train = 5
    model, _parameters, _type = modelgen.generate_models((num_samples_train, num_time_steps, num_channels), 5, 1)[0]
    return model


if __name__ == '__main__':
    unittest.main()
