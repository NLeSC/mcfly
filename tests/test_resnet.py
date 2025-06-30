# -*- coding: utf-8 -*-

import unittest
from mcfly.models import ResNet
from test_modelgen import get_default, generate_train_data

class ResNetSuite(unittest.TestCase):
    """
    Test cases for ResNet models.
    """

    def test_ResNet_starts_with_batchnorm(self):
        """ ResNet models should always start with a batch normalization layer. """
        model_type = ResNet((None, 20, 3), 2)
        model = model_type.create_model(16, 20)

        assert 'BatchNormalization' in str(type(model.layers[1])), 'Wrong layer type.'


    def test_ResNet_first_sandwich_layers(self):
        """ ResNet models should always start with a residual module. """
        model_type = ResNet((None, 20, 3), 2)
        model = model_type.create_model(16, 20)

        assert 'Conv1D' or 'Convolution1D' in str(type(model.layers[2])), 'Wrong layer type.'
        assert 'BatchNormalization' in str(type(model.layers[3])), 'Wrong layer type.'
        assert 'ReLU' in str(type(model.layers[4])), 'Wrong layer type.'


    def test_ResNet_depth(self):
        """ ResNet model should have depth (number of residual modules) as defined by user. """
        depths = 2

        model_type = ResNet((None, 20, 3), 2)
        model = model_type.create_model(16, 20, network_depth=depths)

        add_layers = [str(type(layer)) for layer in model.layers if 'Add' in str(type(layer))]
        assert len(add_layers) == depths, 'Wrong number of residual modules (network depths).'


    def test_ResNet_first_module_dim(self):
        """"The output shape throughout the first residual module should be (None, nr_timesteps, min_filters_number)"""
        min_filters_number = 16

        model_type = ResNet((None, 30, 5), 2)
        model = model_type.create_model(min_filters_number, 20)

        firstConvlayer = model.layers[2]
        firstAddlayer = model.layers[12]
        assert firstConvlayer.output.shape == (None, 30, min_filters_number)
        assert firstAddlayer.output.shape == (None, 30, min_filters_number)

    def test_ResNet_metrics(self):
        """ResNet model should be compiled with the metrics that we give it"""
        metrics = ['accuracy', 'mae']
        x_shape = (None, 20, 3)
        nr_classes = 2
        X_train, y_train = generate_train_data(x_shape, nr_classes)

        model_type = ResNet(x_shape, nr_classes, metrics=metrics)
        model = model_type.create_model(16, 20)
        model.fit(X_train, y_train, epochs=1)

        if "compile_metrics" in model.metrics_names:
            model_metrics = model.metrics[model.metrics_names.index("compile_metrics")].metrics
        else:
            model_metrics = model.metrics
        model_metrics = [metric.name for metric in model_metrics]
        for metric in metrics:
            assert metric in model_metrics

    def test_ResNet_hyperparameters(self):
        """ Network depth from range [4,4] should be 4.
        Maximum kernal size from range [10, 10] should be 10.
        Minimum filter number from range [16, 16] should be 16.  """
        custom_settings = get_default()
        kwargs = {'resnet_min_network_depth': 4,
                  'resnet_max_network_depth': 4,
                  'resnet_min_max_kernel_size': 10,
                  'resnet_max_max_kernel_size': 10,
                  'resnet_min_filters_number': 16,
                  'resnet_max_filters_number': 16}
        # Replace default parameters with input
        for key, value in kwargs.items():
            if key in custom_settings:
                custom_settings[key] = value

        model_type = ResNet(None, None, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()

        assert hyperparams.get('network_depth') == 4, 'Wrong network depth'
        assert hyperparams.get('max_kernel_size') == 10, 'Wrong kernel'
        assert hyperparams.get('min_filters_number') == 16, 'Wrong filter number'


if __name__ == '__main__':
    unittest.main()
