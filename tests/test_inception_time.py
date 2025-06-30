# -*- coding: utf-8 -*-

import unittest
from mcfly.models import InceptionTime
from test_modelgen import get_default, generate_train_data

class InceptionTimeSuite(unittest.TestCase):
    """
    Test cases for InceptionTime models.
    """

    # Tests for InceptionTime model:
    def test_InceptionTime_starts_with_batchnorm(self):
        """ InceptionTime models should always start with a batch normalization layer. """
        model_type = InceptionTime((None, 20, 3), 2)
        model = model_type.create_model(16)

        assert 'BatchNormalization' in str(type(model.layers[1])), 'Wrong layer type.'


    def test_InceptionTime_first_inception_module(self):
        """ Test layers of first inception module. """
        model_type = InceptionTime((None, 20, 3), 2)
        model = model_type.create_model(16)

        assert 'Conv1D' or 'Convolution1D' in str(type(model.layers[2])), 'Wrong layer type.'
        assert 'MaxPooling1D' in str(type(model.layers[3])), 'Wrong layer type.'
        assert 'Concatenate' in str(type(model.layers[8])), 'Wrong layer type.'


    def test_InceptionTime_depth(self):
        """ InceptionTime model should have depth (number of residual modules) as defined by user. """
        depths = 3

        model_type = InceptionTime((None, 20, 3), 2)
        model = model_type.create_model(16, network_depth=depths)

        concat_layers = [str(type(layer)) for layer in model.layers if 'concatenate' in str(type(layer)).lower()]
        assert len(concat_layers) == depths, 'Wrong number of inception modules (network depths).'


    def test_InceptionTime_first_module_dim(self):
        """"The output shape throughout the first residual module should be (None, nr_timesteps, min_filters_number)"""
        min_filters_number = 16

        model_type = InceptionTime((None, 30, 5), 2)
        model = model_type.create_model(min_filters_number)

        secondConvlayer = model.layers[5]
        firstConcatlayer = model.layers[8]
        assert secondConvlayer.output.shape == (None, 30, min_filters_number)
        assert firstConcatlayer.output.shape == (None, 30, min_filters_number * 4)


    def test_InceptionTime_metrics(self):
        """InceptionTime model should be compiled with the metrics that we give it"""
        metrics = ['accuracy', 'mae']
        x_shape = (None, 20, 3)
        nr_classes = 2
        X_train, y_train = generate_train_data(x_shape, nr_classes)

        model_type = InceptionTime(x_shape, nr_classes, metrics=metrics)
        model = model_type.create_model(16)
        model.fit(X_train, y_train)

        if "compile_metrics" in model.metrics_names:
            model_metrics = model.metrics[model.metrics_names.index("compile_metrics")].metrics
        else:
            model_metrics = model.metrics
        model_metrics = [metric.name for metric in model_metrics]
        for metric in metrics:
            assert metric in model_metrics


    def test_InceptionTime_hyperparameters(self):
        """ Network depth from range [5,5] should be 5.
        Maximum kernal size from range [12, 12] should be 12.
        Minimum filter number from range [32, 32] should be 32.  """
        custom_settings = get_default()
        x_shape = (None, 20, 3)
        kwargs = {'IT_min_network_depth': 5,
                  'IT_max_network_depth': 5,
                  'IT_min_max_kernel_size': 10,
                  'IT_max_max_kernel_size': 10,
                  'IT_min_filters_number': 32,
                  'IT_max_filters_number': 32}
        # Replace default parameters with input
        for key, value in kwargs.items():
            if key in custom_settings:
                custom_settings[key] = value

        model_type = InceptionTime(x_shape, None, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()

        assert hyperparams.get('network_depth') == 5, 'Wrong network depth'
        assert hyperparams.get('max_kernel_size') == 10, 'Wrong kernel'
        assert hyperparams.get('filters_number') == 32, 'Wrong filter number'


if __name__ == '__main__':
    unittest.main()
