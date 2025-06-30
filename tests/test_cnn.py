# -*- coding: utf-8 -*-
import unittest
from mcfly.models import CNN
from test_modelgen import get_default, generate_train_data


class CNNSuite(unittest.TestCase):
    """
    Test cases for CNN models.
    """

    def test_cnn_starts_with_batchnorm(self):
        """ CNN models should always start with a batch normalization layer. """
        model_type = CNN((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "fc_hidden_nodes": 100})
        assert 'BatchNormalization' in str(type(model.layers[0])), 'Wrong layer type.'


    def test_cnn_fc_nodes(self):
        """ CNN model should have number of dense nodes defined by user. """
        fc_hidden_nodes = 101
        model_type = CNN((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "fc_hidden_nodes": fc_hidden_nodes})

        dense_layer = [layer for layer in model.layers if 'Dense' in str(layer)][0]
        assert dense_layer.output.shape[1] == fc_hidden_nodes, 'Wrong number of fc nodes.'


    def test_cnn_batchnorm_dim(self):
        """"The output shape of the batchnorm should be (None, nr_timesteps, nr_filters)"""
        model_type = CNN((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "fc_hidden_nodes": 100})

        batchnormlay = model.layers[2]
        assert batchnormlay.output.shape == (None, 20, 32)


    def test_cnn_enough_batchnorm(self):
        """CNN model should contain as many batch norm layers as it has activations layers"""
        model_type = CNN((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "fc_hidden_nodes": 100})

        batch_norm_layers = len([layer for layer in model.layers if 'BatchNormalization' in str(layer)])
        activation_layers = len([layer for layer in model.layers if 'Activation' in str(layer)])
        assert batch_norm_layers == activation_layers


    def test_cnn_metrics(self):
        """CNN model should be compiled with the metrics that we give it"""
        metrics = ['accuracy', 'mae']
        x_shape = (None, 20, 3)
        nr_classes = 2
        X_train, y_train = generate_train_data(x_shape, nr_classes)

        model_type = CNN(x_shape, nr_classes, metrics=metrics)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "fc_hidden_nodes": 100})
        model.fit(X_train, y_train, epochs=1)

        if "compile_metrics" in model.metrics_names:
            model_metrics = model.metrics[model.metrics_names.index("compile_metrics")].metrics
        else:
            model_metrics = model.metrics
        model_metrics = [metric.name for metric in model_metrics]
        for metric in metrics:
            assert metric in model_metrics


    def test_CNN_hyperparameters_nrlayers(self):
        """ Number of Conv layers from range [4, 4] should be 4. """
        custom_settings = get_default()
        kwargs = {'cnn_min_layers': 4,
                  'cnn_max_layers': 4}
        # Replace default parameters with input
        for key, value in kwargs.items():
            if key in custom_settings:
                custom_settings[key] = value

        model_type = CNN(None, None, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()

        assert len(hyperparams.get('filters')) == 4


    def test_CNN_hyperparameters_fcnodes(self):
        """ Number of fc nodes from range [123, 123] should be 123. """
        custom_settings = get_default()
        kwargs = {'cnn_min_fc_nodes': 123,
                  'cnn_max_fc_nodes': 123}
        # Replace default parameters with input
        for key, value in kwargs.items():
            if key in custom_settings:
                custom_settings[key] = value

        model_type = CNN(None, None, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()

        assert hyperparams.get('fc_hidden_nodes') == 123



if __name__ == '__main__':
    unittest.main()
