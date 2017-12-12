from mcfly import find_architecture
import numpy as np
from keras.utils.np_utils import to_categorical
import os
import unittest
import pytest

try:
    import noodles
    from mcfly.storage import serial_registry
except ImportError:
    has_noodles = False
else:
    has_noodles = True


@pytest.mark.skipif(
        not has_noodles,
        reason="This test needs Noodles.")
class NoodlesSuite(unittest.TestCase):
    """Basic test cases."""

    def test_find_best_architecture(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
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

        def run(wf):
            return noodles.run_process(wf, n_processes=4, registry=serial_registry)

        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            X_train, y_train, X_val, y_val, verbose=False, subset_size=10,
            number_of_models=1, nr_epochs=1, use_noodles=run)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        assert 1 >= knn_acc >= 0

    def train_models_on_samples_empty(self):
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

        def run(wf):
            return noodles.run_process(wf, n_processes=4, registry=serial_registry)

        histories, val_metrics, val_losses = \
            find_architecture.train_models_on_samples(
                X_train, y_train, X_val, y_val, [],
                nr_epochs=1, subset_size=10, verbose=False,
                outputfile=None, early_stopping=False,
                batch_size=20, metric='accuracy', use_noodles=run)
        assert len(histories) == 0

    def setUp(self):
        np.random.seed(1234)


if __name__ == '__main__':
    unittest.main()
