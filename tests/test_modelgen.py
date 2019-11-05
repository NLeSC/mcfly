from mcfly import modelgen
import numpy as np

import unittest


class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def test_regularization_is_float(self):
        """ Regularization should be a float. """
        reg = modelgen.get_regularization(0, 5)
        assert type(reg) == np.float

    def test_regularization_0size_interval(self):
        """ Regularization from zero size interval [2,2] should be 10^-2. """
        reg = modelgen.get_regularization(2, 2)
        assert reg == 0.01

    def test_base_hyper_parameters_reg(self):
        """ Base hyper parameter set should contain regularization. """
        hyper_parameter_set = modelgen.generate_base_hyper_parameter_set()
        assert 'regularization_rate' in hyper_parameter_set.keys()

    def test_cnn_starts_with_batchnorm(self):
        """ CNN models should always start with a batch normalization layer. """
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
        assert 'BatchNormalization' in str(type(model.layers[0])), 'Wrong layer type.'

    def test_cnn_fc_nodes(self):
        """ CNN model should have number of dense nodes defined by user. """
        fc_hidden_nodes = 101
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], fc_hidden_nodes)
        dense_layer = [l for l in model.layers if 'Dense' in str(l)][0]
        assert dense_layer.output_shape[1] == fc_hidden_nodes, 'Wrong number of fc nodes.'

    def test_cnn_batchnorm_dim(self):
        """"The output shape of the batchnorm should be (None, nr_timesteps, nr_filters)"""
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
        batchnormlay = model.layers[2]
        assert batchnormlay.output_shape == (None, 20, 32)

    def test_deepconvlstm_batchnorm_dim(self):
        """The output shape of the batchnorm should be (None, nr_timesteps, nr_channels, nr_filters)"""
        model = modelgen.generate_DeepConvLSTM_model((None, 20, 3), 2, [32, 32], [32, 32])
        batchnormlay = model.layers[3]
        assert batchnormlay.output_shape == (None, 20, 3, 32)

    def test_deepconvlstm_enough_batchnorm(self):
        """LSTM model should contain as many batch norm layers as it has activations layers"""
        model = modelgen.generate_DeepConvLSTM_model(
            (None, 20, 3), 2, [32, 32, 32], [32, 32, 32])
        batch_norm_layers = len([l for l in model.layers if 'BatchNormalization' in str(l)])
        activation_layers = len([l for l in model.layers if 'Activation' in str(l)])
        assert batch_norm_layers == activation_layers

    def test_cnn_enough_batchnorm(self):
        """CNN model should contain as many batch norm layers as it has activations layers"""
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
        batch_norm_layers = len([l for l in model.layers if 'BatchNormalization' in str(l)])
        activation_layers = len([l for l in model.layers if 'Activation' in str(l)])
        assert batch_norm_layers == activation_layers

    def test_cnn_metrics(self):
        """CNN model should be compiled with the metrics that we give it"""
        metrics = ['accuracy', 'mae']
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100, metrics=metrics)
        model_metrics = [m.name for m in model.metrics]
        assert model_metrics == metrics or model_metrics == ['acc', 'mean_absolute_error']

    def test_CNN_hyperparameters_nrlayers(self):
        """ Number of Conv layers from range [4, 4] should be 4. """
        hyperparams = modelgen.generate_CNN_hyperparameter_set(min_layers=4, max_layers=4)
        assert len(hyperparams.get('filters')) == 4

    def test_CNN_hyperparameters_nrlayers(self):
        """ Number of fc nodes from range [123, 123] should be 123. """
        hyperparams = modelgen.generate_CNN_hyperparameter_set(min_fc_nodes=123, max_fc_nodes=123)
        assert hyperparams.get('fc_hidden_nodes') == 123

    def test_DeepConvLSTM_hyperparameters_nrconvlayers(self):
        """ Number of Conv layers from range [4, 4] should be 4. """
        hyperparams = modelgen.generate_DeepConvLSTM_hyperparameter_set(min_conv_layers=4, max_conv_layers=4)
        assert len(hyperparams.get('filters')) == 4

    def test_deepconvlstm_starts_with_batchnorm(self):
        """ DeepConvLSTM models should always start with a batch normalization layer. """
        model = modelgen.generate_DeepConvLSTM_model((None, 20, 3), 2, [32, 32], [32, 32])
        assert 'BatchNormalization' in str(type(model.layers[0])), 'Wrong layer type.'

    def test_generate_models_metrics(self):
        models = modelgen.generate_models((None, 20, 3), 2)
        model, hyperparams, modeltype = models[0]
        metrics = [m.name for m in model.metrics]
        assert metrics == ['accuracy'] or metrics == ['acc']

    def setUp(self):
        np.random.seed(1234)


if __name__ == '__main__':
    unittest.main()
