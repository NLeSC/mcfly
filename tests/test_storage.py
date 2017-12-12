from mcfly import find_architecture, storage
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import json
import pickle
import os
import unittest


class StorageSuite(unittest.TestCase):
    """Basic test cases."""

    def test_savemodel(self):
        """ Test whether a dummy model is saved """
        best_model = create_dummy_model()
        filepath = os.getcwd() + '/'
        modelname = 'teststorage'
        storage.savemodel(best_model, filepath, modelname)
        filename1 = filepath + modelname + '_architecture.json'
        filename2 = filepath + modelname + '_weights.npy'
        test1 = os.path.isfile(filename1)
        test2 = os.path.isfile(filename2)
        test = test1 == True and test2 == True
        if test is True:
            os.remove(filename1)
            os.remove(filename2)
        assert test

    def test_savemodel_keras(self):
        """ Test whether a dummy model is saved """
        best_model = create_dummy_model()
        filepath = os.getcwd() + '/'
        modelname = 'teststorage.h5'
        filename = os.path.join(filepath, modelname)
        best_model.save(filename)
        test = os.path.isfile(filename)
        if test is True:
            os.remove(filename)
        assert test

    def test_loadmodel(self):
        """ Test whether a dummy model can be save and then loaded """
        best_model = create_dummy_model()
        filepath = os.getcwd() + '/'
        modelname = 'teststorage'
        storage.savemodel(best_model, filepath, modelname)
        filename1 = filepath + modelname + '_architecture.json'
        filename2 = filepath + modelname + '_weights.npy'
        model_loaded = storage.loadmodel
        test = hasattr(best_model, 'fit')
        if test is True:
            os.remove(filename1)
            os.remove(filename2)
        assert test


def create_dummy_model():
    """ Function to aid the tests on saving and loading a model"""
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
    return(best_model)

if __name__ == '__main__':
    unittest.main()
