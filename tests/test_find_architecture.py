import pytest
from pytest import approx, raises
import unittest
import math
import json
import numpy as np
import keras
from keras.utils import to_categorical, Sequence
from test_tools import safe_remove

from mcfly import find_architecture
from mcfly.models import CNN
from mcfly.modelgen import Task
from mcfly.keras_dataset import NumpyKerasDataset
from test_modelgen import get_default as get_default_settings


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        return batch_x, batch_y


def _create_2_class_labeled_dataset(num_samples_class_a, num_samples_class_b):
    X = _create_2_class_noisy_data(num_samples_class_a, num_samples_class_b)
    y = _create_2_class_labels(num_samples_class_a, num_samples_class_b)
    return X, y


def _create_2_class_noisy_data(num_samples_class_a, num_samples_class_b):
    num_channels = 2
    num_time_steps = 100
    data_class_a = np.zeros((num_samples_class_a, num_time_steps, num_channels))
    data_class_b = np.ones((num_samples_class_b, num_time_steps, num_channels))
    signal = np.vstack((data_class_a, data_class_b))
    noise = 0.1 * np.random.randn(signal.shape[0], signal.shape[1], signal.shape[2])
    return signal + noise


def _create_2_class_labels(num_samples_class_a, num_samples_class_b):
    labels_class_a = np.zeros(num_samples_class_a)
    labels_class_b = np.ones(num_samples_class_b)
    return to_categorical(np.hstack((labels_class_a, labels_class_b)))


def _create_regression_dataset(num_samples, y_dims=1):
    num_channels = 2
    num_time_steps = 100

    X = np.random.uniform(-1.0, 1.0, size=(num_samples, num_time_steps, num_channels))
    y = np.random.uniform(-1.0, 1.0, size=(num_samples, y_dims))

    return X, y


class FindArchitectureBasicSuite(unittest.TestCase):
    classification_train_dataset = _create_2_class_labeled_dataset(5, 5)
    classification_val_dataset = _create_2_class_labeled_dataset(3, 3)

    regression_train_dataset = _create_regression_dataset(10)
    regression_val_dataset = _create_regression_dataset(6)

    batch_size = 5

    def test_kNN_accuracy_1(self):
        """
        The accuracy for this single-point dataset should be 1.
        """
        X_train = np.array([[[1]], [[0]]])
        y_train = np.array([[1, 0], [0, 1]])
        X_val = np.array([[[0.9]]])
        y_val = np.array([[1, 0]])

        acc = find_architecture.kNN_performance(
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

        acc = find_architecture.kNN_performance(
            X_train, y_train, X_val, y_val, k=1)
        assert acc == approx(0)

    def test_kNN_mse_above_0(self):
        """
        The mean squared error for this single-point dataset should be above 0.
        """
        X_train = np.array([[[1]], [[0]]])
        y_train = np.array([[1,], [0,]])
        X_val = np.array([[[0.9]]])
        y_val = np.array([[0,]])

        mse = find_architecture.kNN_performance(
            X_train, y_train, X_val, y_val, k=1, task=Task.regression)
        assert mse > 0

    def test_kNN_mse_0(self):
        """
        The mean squared error for this single-point dataset should be 0.
        """
        X_train = np.array([[[1]], [[0]]])
        y_train = np.array([[1,], [0,]])
        X_val = np.array([[[1]]])
        y_val = np.array([[1,]])

        mse = find_architecture.kNN_performance(
            X_train, y_train, X_val, y_val, k=1, task=Task.regression)
        assert mse == approx(0)

    def test_find_best_architecture_classification(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset
        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            X_train, y_train, X_val, y_val, verbose=False, subset_size=10,
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        assert 1 >= knn_acc >= 0

    def test_find_best_architecture_classification_with_numpy_dataset(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = NumpyKerasDataset(X_train, y_train, self.batch_size)
        data_val = NumpyKerasDataset(X_val, y_val, self.batch_size)

        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            data_train, None, data_val, None, verbose=False, subset_size=None,
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        self.assertIsNone(knn_acc)

    @unittest.skipIf(keras.backend.backend() != "tensorflow", reason="requires keras backend tensorflow")
    @pytest.mark.tensorflow
    def test_find_best_architecture_classification_with_tensorflow_dataset(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
        assert keras.backend.backend() == "tensorflow", "Unexpected keras backend."
        import tensorflow as tf

        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(self.batch_size)
        data_val = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)).batch(self.batch_size)

        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            data_train, None, data_val, None, verbose=False, subset_size=None,
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        self.assertIsNone(knn_acc)

    @unittest.skipIf(keras.backend.backend() != "torch", reason="requires keras backend torch")
    @pytest.mark.torch
    def test_find_best_architecture_classification_with_torch_dataset(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
        assert keras.backend.backend() == "torch", "Unexpected keras backend."
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size
        )
        data_val = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size
        )

        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            data_train, None, data_val, None, verbose=False, subset_size=None,
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        self.assertIsNone(knn_acc)

    def test_find_best_architecture_classification_with_generator(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = DataGenerator(X_train, y_train, self.batch_size)
        data_val = DataGenerator(X_val, y_val, self.batch_size)

        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            data_train, None, data_val, None, verbose=False, subset_size=None,
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        self.assertIsNone(knn_acc)

    def test_find_best_architecture_classification_non_default_metric(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn accuracy."""
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset
        best_model, best_params, best_model_type, knn_acc = find_architecture.find_best_architecture(
            X_train, y_train, X_val, y_val, verbose=False, subset_size=10, metric='categorical_accuracy',
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        self.assertIsNone(knn_acc)

    def test_find_best_architecture_regression(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn mean squared error."""
        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset
        best_model, best_params, best_model_type, knn_mse = find_architecture.find_best_architecture(
            X_train, y_train, X_val, y_val, verbose=False, subset_size=10,
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        assert knn_mse >= 0

    def test_find_best_architecture_regression_non_default_metric(self):
        """ Find_best_architecture should return a single model, parameters, type and valid knn mean squared error."""
        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset
        best_model, best_params, best_model_type, knn_mse = find_architecture.find_best_architecture(
            X_train, y_train, X_val, y_val, verbose=False, subset_size=10, metric='mean_absolute_error',
            number_of_models=1, nr_epochs=1)
        assert hasattr(best_model, 'fit')
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_model_type)
        self.assertIsNone(knn_mse)

    # %TODO add test with metric other than accuracy
    # TODO: Is this a test? It's not set up as one
    def train_models_on_samples_empty(self):
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        histories, _, _ = \
            find_architecture.train_models_on_samples(
                X_train, y_train, X_val, y_val, [],
                nr_epochs=1, subset_size=10, verbose=False,
                outputfile=None,
                batch_size=self.batch_size, metric='accuracy')
        assert len(histories) == 0

    def test_train_models_on_samples_with_x_and_y(self):
        """
        Model should be able to train using separated x and y values
        """
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        custom_settings = get_default_settings()
        model_type = CNN(X_train.shape, 2, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()
        model = model_type.create_model(**hyperparams)
        models = [(model, hyperparams, "CNN")]

        histories, _, _ = \
            find_architecture.train_models_on_samples(
                X_train, y_train, X_val, y_val, models,
                nr_epochs=1, subset_size=10, verbose=False,
                outputfile=None, early_stopping_patience='auto',
                batch_size=self.batch_size)
        assert len(histories) == 1

    def test_train_models_on_samples_with_numpy_dataset(self):
        """
        Model should be able to train using a dataset as an input
        """
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = NumpyKerasDataset(X_train, y_train, self.batch_size)
        data_val = NumpyKerasDataset(X_val, y_val, self.batch_size)

        custom_settings = get_default_settings()
        model_type = CNN(X_train.shape, 2, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()
        model = model_type.create_model(**hyperparams)
        models = [(model, hyperparams, "CNN")]

        histories, _, _ = \
            find_architecture.train_models_on_samples(
                data_train, None, data_val, None, models,
                nr_epochs=1, subset_size=None, verbose=False,
                outputfile=None, early_stopping_patience='auto',
                batch_size=self.batch_size)
        assert len(histories) == 1

    @unittest.skipIf(keras.backend.backend() != "tensorflow", reason="requires keras backend tensorflow")
    @pytest.mark.tensorflow
    def test_train_models_on_samples_with_tensorflow_dataset(self):
        """
        Model should be able to train using a dataset as an input
        """
        assert keras.backend.backend() == "tensorflow", "Unexpected keras backend."
        import tensorflow as tf

        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(self.batch_size)
        data_val = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)).batch(self.batch_size)

        custom_settings = get_default_settings()
        model_type = CNN(X_train.shape, 2, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()
        model = model_type.create_model(**hyperparams)
        models = [(model, hyperparams, "CNN")]

        histories, _, _ = \
            find_architecture.train_models_on_samples(
                data_train, None, data_val, None, models,
                nr_epochs=1, subset_size=None, verbose=False,
                outputfile=None, early_stopping_patience='auto',
                batch_size=self.batch_size)
        assert len(histories) == 1

    @unittest.skipIf(keras.backend.backend() != "torch", reason="requires keras backend torch")
    @pytest.mark.torch
    def test_train_models_on_samples_with_torch_dataset(self):
        """
        Model should be able to train using a dataset as an input
        """
        assert keras.backend.backend() == "torch", "Unexpected keras backend."
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size
        )
        data_val = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size
        )

        custom_settings = get_default_settings()
        model_type = CNN(X_train.shape, 2, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()
        model = model_type.create_model(**hyperparams)
        models = [(model, hyperparams, "CNN")]

        histories, _, _ = \
            find_architecture.train_models_on_samples(
                data_train, None, data_val, None, models,
                nr_epochs=1, subset_size=None, verbose=False,
                outputfile=None, early_stopping_patience='auto',
                batch_size=self.batch_size)
        assert len(histories) == 1

    def test_train_models_on_samples_with_generators(self):
        """
        Model should be able to train using a generator as an input
        """
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = DataGenerator(X_train, y_train, self.batch_size)
        data_val = DataGenerator(X_val, y_val, self.batch_size)

        custom_settings = get_default_settings()
        model_type = CNN(X_train.shape, 2, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()
        model = model_type.create_model(**hyperparams)
        models = [(model, hyperparams, "CNN")]

        histories, _, _ = \
            find_architecture.train_models_on_samples(
                data_train, None, data_val, None, models,
                nr_epochs=1, subset_size=None, verbose=False,
                outputfile=None, early_stopping_patience='auto',
                batch_size=self.batch_size)
        assert len(histories) == 1

    def setUp(self):
        np.random.seed(1234)


class TaskInferenceSuite(unittest.TestCase):
    classification_train_dataset = _create_2_class_labeled_dataset(5, 5)
    classification_val_dataset = _create_2_class_labeled_dataset(3, 3)

    regression_train_dataset = _create_regression_dataset(10)
    regression_val_dataset = _create_regression_dataset(6)

    batch_size = 5


    def test_infer_task_from_y_different_dtype(self):
        _, y_train = self.classification_train_dataset
        _, y_val = self.regression_val_dataset

        with raises(ValueError, match="Both 'y_train' and 'y_val' must be one-hot encoding or continuous"):
            find_architecture._infer_task_from_y(y_train, y_val)


    def test_infer_task_from_y_classification(self):
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        task = find_architecture._infer_task(X_train, X_val, y_train, y_val)

        assert task == Task.classification


    def test_infer_task_from_y_regression(self):
        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset

        task = find_architecture._infer_task(X_train, X_val, y_train, y_val)

        assert task == Task.regression


    def test_infer_task_generator_classification(self):
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = DataGenerator(X_train, y_train, self.batch_size)
        data_val = DataGenerator(X_val, y_val, self.batch_size)

        task = find_architecture._infer_task(data_train, data_val, None, None)
        
        assert task == Task.classification


    def test_infer_task_generator_regression(self):
        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset

        data_train = DataGenerator(X_train, y_train, self.batch_size)
        data_val = DataGenerator(X_val, y_val, self.batch_size)

        task = find_architecture._infer_task(data_train, data_val, None, None)
        
        assert task == Task.regression


    def test_infer_task_numpy_dataset_classification(self):
        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = NumpyKerasDataset(X_train, y_train, self.batch_size)
        data_val = NumpyKerasDataset(X_val, y_val, self.batch_size)

        task = find_architecture._infer_task(data_train, data_val, None, None)

        assert task == Task.classification

    @unittest.skipIf(keras.backend.backend() != "tensorflow", reason="requires keras backend tensorflow")
    @pytest.mark.tensorflow
    def test_infer_task_tensorflow_dataset_classification(self):
        assert keras.backend.backend() == "tensorflow", "Unexpected keras backend."
        import tensorflow as tf

        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(self.batch_size)
        data_val = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)).batch(self.batch_size)

        task = find_architecture._infer_task(data_train, data_val, None, None)

        assert task == Task.classification

    @unittest.skipIf(keras.backend.backend() != "torch", reason="requires keras backend torch")
    @pytest.mark.torch
    def test_infer_task_torch_dataset_classification(self):
        assert keras.backend.backend() == "torch", "Unexpected keras backend."
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_train, y_train = self.classification_train_dataset
        X_val, y_val = self.classification_val_dataset

        data_train = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size
        )
        data_val = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size
        )

        task = find_architecture._infer_task(data_train, data_val, None, None)

        assert task == Task.classification


    def test_infer_task_numpy_dataset_regression(self):
        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset

        data_train = NumpyKerasDataset(X_train, y_train, self.batch_size)
        data_val = NumpyKerasDataset(X_val, y_val, self.batch_size)

        task = find_architecture._infer_task(data_train, data_val, None, None)

        assert task == Task.regression

    @unittest.skipIf(keras.backend.backend() != "tensorflow", reason="requires keras backend tensorflow")
    @pytest.mark.tensorflow
    def test_infer_task_tensorflow_dataset_regression(self):
        assert keras.backend.backend() == "tensorflow", "Unexpected keras backend."
        import tensorflow as tf

        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset

        data_train = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(self.batch_size)
        data_val = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)).batch(self.batch_size)

        task = find_architecture._infer_task(data_train, data_val, None, None)

        assert task == Task.regression

    @unittest.skipIf(keras.backend.backend() != "torch", reason="requires keras backend torch")
    @pytest.mark.torch
    def test_infer_task_torch_dataset_regression(self):
        assert keras.backend.backend() == "torch", "Unexpected keras backend."
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        X_train, y_train = self.regression_train_dataset
        X_val, y_val = self.regression_val_dataset

        data_train = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size
        )
        data_val = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size
        )

        task = find_architecture._infer_task(data_train, data_val, None, None)

        assert task == Task.regression


class MetricNamingSuite(unittest.TestCase):
    @staticmethod
    def test_get_metric_name_accuracy():
        metric_name = find_architecture._get_metric_name('accuracy')
        assert metric_name == 'accuracy'

    @staticmethod
    def test_get_metric_name_acc():
        metric_name = find_architecture._get_metric_name('acc')
        assert metric_name == 'accuracy'

    @staticmethod
    def test_get_metric_name_myfunc():
        def myfunc(a, b):
            return None

        metric_name = find_architecture._get_metric_name(myfunc)
        assert metric_name == 'myfunc'

    @staticmethod
    def test_val_accuracy_get_from_history_acc():
        history_history = {'val_acc': 'val_accuracy'}
        result = find_architecture._get_from_history('val_accuracy', history_history)
        assert result == 'val_accuracy'

    @staticmethod
    def test_val_accuracy_get_from_history_accuracy():
        history_history = {'val_accuracy': 'val_accuracy'}
        result = find_architecture._get_from_history('val_accuracy', history_history)
        assert result == 'val_accuracy'

    @staticmethod
    def test_val_loss_get_from_history_accuracy():
        history_history = {'val_loss': 'val_loss'}
        result = find_architecture._get_from_history('val_loss', history_history)
        assert result == 'val_loss'

    @staticmethod
    def test_val_accuracy_get_from_history_none_raise():
        history_history = {}
        with raises(KeyError):
            find_architecture._get_from_history('val_accuracy', history_history)

    @staticmethod
    def test_accuracy_get_from_history_acc():
        history_history = {'acc': 'accuracy'}
        result = find_architecture._get_from_history('accuracy', history_history)
        assert result == 'accuracy'

    @staticmethod
    def test_accuracy_get_from_history_accuracy():
        history_history = {'accuracy': 'accuracy'}
        result = find_architecture._get_from_history('accuracy', history_history)
        assert result == 'accuracy'

    @staticmethod
    def test_accuracy_get_from_history_none_raise():
        history_history = {}
        with raises(KeyError):
            find_architecture._get_from_history('accuracy', history_history)


class HistoryStoringSuite(unittest.TestCase):
    def test_store_train_history_as_json_contains_expected_attributes(self):
        """The code should produce a json file with a number of expected attributes, used by visualization."""
        self._write_test_history_file(self.history_file_path)

        expected_attributes = ['metrics', 'modeltype', 'regularization_rate', 'learning_rate']
        log = self._load_history_and_assert_is_list(self.history_file_path)
        for model_log in log:
            for attribute in expected_attributes:
                assert attribute in model_log

    def test_store_train_history_as_json_metrics_is_dict(self):
        """The log for every model should contain a dict allowing for multiple metrics."""
        self._write_test_history_file(self.history_file_path)

        log = self._load_history_and_assert_is_list(self.history_file_path)
        for model_log in log:
            assert isinstance(model_log['metrics'], dict)

    @staticmethod
    def _write_test_history_file(history_file_path):
        params = {'fc_hidden_nodes': 1,
                  'learning_rate': 1,
                  'regularization_rate': 0,
                  'filters': np.array([1, 1]),
                  'lstm_dims': np.array([1, 1])
                  }
        history = {'loss': [1, 1], 'accuracy': [np.float64(0), np.float64(0)],
                   'val_loss': [np.float64(1), np.float64(1)], 'val_accuracy': [np.float64(0), np.float64(0)],
                   'my_own_custom_metric': [np.float64(0), np.float64(0)]}
        model_type = 'ABC'
        find_architecture.store_train_hist_as_json(params, model_type, history, history_file_path)

    @staticmethod
    def _load_history_and_assert_is_list(history_file_path):
        """ The log contains a list of models with their history. Top level should therefore be type list."""
        with open(history_file_path) as f:
            log = json.load(f)
        # In case any assertion fails, we want to see the complete log printed to console.
        print(log)
        assert isinstance(log, list)
        return log

    def setUp(self):
        self.history_file_path = '.generated_models_history_for_storing_test.json'
        safe_remove(self.history_file_path)

    def tearDown(self):
        safe_remove(self.history_file_path)


if __name__ == '__main__':
    unittest.main()
