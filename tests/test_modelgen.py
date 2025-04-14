import numpy as np
import unittest
import pytest
from keras.models import Model
from mcfly import modelgen
from mcfly.models import ResNet


def get_default():
    """ "Define mcflu default parameters as dictionary. """
    settings = {'metrics': ['accuracy'],
                'model_types': ['CNN', 'DeepConvLSTM', 'ResNet', 'InceptionTime'],
                'cnn_min_layers': 1,
                'cnn_max_layers': 10,
                'cnn_min_filters': 10,
                'cnn_max_filters': 100,
                'cnn_min_fc_nodes': 10,
                'cnn_max_fc_nodes': 2000,
                'deepconvlstm_min_conv_layers': 1,
                'deepconvlstm_max_conv_layers': 10,
                'deepconvlstm_min_conv_filters': 10,
                'deepconvlstm_max_conv_filters': 100,
                'deepconvlstm_min_lstm_layers': 1,
                'deepconvlstm_max_lstm_layers': 5,
                'deepconvlstm_min_lstm_dims': 10,
                'deepconvlstm_max_lstm_dims': 100,
                'resnet_min_network_depth': 2,
                'resnet_max_network_depth': 5,
                'resnet_min_filters_number': 32,
                'resnet_max_filters_number': 128,
                'resnet_min_max_kernel_size': 8,
                'resnet_max_max_kernel_size': 32,
                'IT_min_network_depth': 3,
                'IT_max_network_depth': 6,
                'IT_min_filters_number': 32,
                'IT_max_filters_number': 96,
                'IT_min_max_kernel_size': 10,
                'IT_max_max_kernel_size': 100,
                'low_lr': 1,
                'high_lr': 4,
                'low_reg': 1,
                'high_reg': 4}
    return settings


def generate_train_data(x_shape, nr_classes):
    X_train = np.random.rand(1, *x_shape[1:])
    y_train = np.random.randint(0, 1, size=(1, nr_classes))
    return X_train, y_train


class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    # Tests for general mcfly functionality:
    def test_generate_models_metrics(self):
        """ Test if correct number of models is generated and if metrics is correct. """
        x_shape = (None, 20, 10)
        nr_classes = 2
        X_train, y_train = generate_train_data(x_shape, nr_classes)
        n_models = 5

        models = modelgen.generate_models(x_shape, nr_classes, n_models)
        for model in models:
            model[0].fit(X_train, y_train, epochs = 1)

        model, _, modeltype = models[0]
        if "compile_metrics" in model.metrics_names:
            model_metrics = model.metrics[model.metrics_names.index("compile_metrics")].metrics
        else:
            model_metrics = model.metrics
        model_metrics = [metric.name for metric in model_metrics]
        assert "accuracy" in model_metrics, "Not found accuracy for model {}. Found {}".format(
            modeltype, model_metrics)
        assert len(models) == n_models, "Expecting {} models, found {} models".format(
            n_models, len(models))

    def test_generate_models_pass_model_object(self):
        """ Test if model class can be passed as model_types input."""
        x_shape = (None, 20, 10)
        nr_classes = 2
        n_models = 4

        models = modelgen.generate_models(x_shape, nr_classes, n_models,
                                          model_types=['CNN', ResNet])
        created_model_names = list({x[2] for x in models})
        created_model_names.sort()
        assert len(models) == 4, "Expected number of models to be 4"
        assert created_model_names == ["CNN", "ResNet"], "Expected different model names."
        for model in models:
            assert isinstance(model[0], Model), "Expected keras model."

    def test_generate_models_exception(self):
        """ Test expected generate_models exception."""
        x_shape = (None, 20, 10)
        nr_classes = 2
        n_models = 2

        with pytest.raises(NameError, match="Unknown model name, 'wrong_entry'."):
            _ = modelgen.generate_models(x_shape, nr_classes, n_models,
                                         model_types=['CNN', "wrong_entry"])

    def setUp(self):
        np.random.seed(1234)


if __name__ == '__main__':
    unittest.main()
