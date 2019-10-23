from mcfly import find_architecture
import numpy as np
from pytest import approx
from tensorflow.keras.utils import to_categorical
import os
import unittest

from test_tools import save_remove


class FindArchitectureSuite(unittest.TestCase):
    """Basic test cases."""

    def test_kNN_accuracy_1(self):
        """
        The accuracy for this single-point dataset should be 1.
        """
        X_train = np.array([[[1]], [[0]]])
        y_train = np.array([[1, 0], [0, 1]])
        X_val = np.array([[[0.9]]])
        y_val = np.array([[1, 0]])

        acc = find_architecture.kNN_accuracy(
            X_train, y_train, X_val, y_val, k=1)
        assert acc == approx(1.0)

    def test_kNN_accuracy_0(self):
        """
        The accuracy for this single-point dataset should be 0.
        """
        X_train = np.array([[[1]], [[0]]])
        y_train = np.array([[1, 0], [0, 1]])
        X_val = np.array([[[0.9]]])
        y_val = np.array([[0, 1]])

        acc = find_architecture.kNN_accuracy(
            X_train, y_train, X_val, y_val, k=1)
        assert acc == approx(0)

    def test_find_best_architecture(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
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
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        assert 1 >= knn_acc >= 0

    def train_models_on_samples_empty(self):
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

        histories, val_metrics, val_losses = \
            find_architecture.train_models_on_samples(
                X_train, y_train, X_val, y_val, [],
                nr_epochs=1, subset_size=10, verbose=False,
                outputfile=None, early_stopping=False,
                batch_size=20, metric='accuracy')
        assert len(histories) == 0

    def test_get_metricname_myfunc(self):
        def myfunc(a, b):
            return None

        metric_name = find_architecture.get_metric_name(myfunc)
        assert metric_name == 'myfunc'

    def setUp(self):
        np.random.seed(1234)


class HistoryStoringSuite(unittest.TestCase):
    def test_store_train_history_as_json(self):
        """The code should produce a json file."""
        params = {'fc_hidden_nodes': 1,
                  'learning_rate': 1,
                  'regularization_rate': 0,
                  'filters': np.array([1, 1]),
                  'lstm_dims': np.array([1, 1])
                  }
        history = {'loss': [1, 1], 'accuracy': [np.float64(0), np.float32(0)],
                   'val_loss': [np.float(1), np.float(1)], 'val_accuracy': [np.float64(0), np.float64(0)]}
        model_type = 'ABC'

        find_architecture.store_train_hist_as_json(params, model_type, history, self.history_file_path)
        assert os.path.isfile(self.history_file_path)

    def setUp(self):
        self.history_file_path = '.generated_models_history_for_storing_test.json'
        save_remove(self.history_file_path)

    def tearDown(self):
        save_remove(self.history_file_path)


if __name__ == '__main__':
    unittest.main()
