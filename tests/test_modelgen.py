from mcfly import modelgen
import numpy as np
from nose.tools import assert_equal, assert_equals

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
        assert_equal(reg, 0.01)

    def test_base_hyper_parameters_reg(self):
        """ Base hyper parameter set should contain regularization. """
        hyper_parameter_set = modelgen.generate_base_hyper_parameter_set()
        assert 'regularization_rate' in hyper_parameter_set.keys()

    def test_cnn_starts_with_batchnorm(self):
        """ CNN models should always start with a batch normalization layer. """
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
        assert_equal(str(type(model.layers[0])), "<class 'keras.layers.normalization.BatchNormalization'>", 'Wrong layer type.')

    def test_cnn_batchnorm_dim(self):
        "The output shape of the batchnorm should be (None, nr_timesteps, nr_filters)"
        model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
        batchnormlay = model.layers[2]
        assert_equal(batchnormlay.output_shape, (None, 20, 32))

    def test_deepconvlstm_batchnorm_dim(self):
        "The output shape of the batchnorm should be (None, nr_filters, nr_timesteps, nr_channels)"
        model = modelgen.generate_DeepConvLSTM_model((None, 20, 3), 2, [32, 32], [32, 32])
        batchnormlay = model.layers[3]
        assert_equal(batchnormlay.output_shape, (None, 32, 20, 3))

    def test_CNN_hyperparameters_nrlayers(self):
        """ Number of Conv layers from range [4, 4] should be 4. """
        hyperparams = modelgen.generate_CNN_hyperparameter_set(min_layers=4, max_layers=4)
        assert_equal(len(hyperparams.get('filters')), 4)

    def test_DeepConvLSTM_hyperparameters_nrconvlayers(self):
        """ Number of Conv layers from range [4, 4] should be 4. """
        hyperparams = modelgen.generate_DeepConvLSTM_hyperparameter_set(min_conv_layers=4, max_conv_layers=4)
        assert_equal(len(hyperparams.get('filters')), 4)

    def test_deepconvlstm_starts_with_batchnorm(self):
        """ DeepConvLSTM models should always start with a batch normalization layer. """
        model = modelgen.generate_DeepConvLSTM_model((None, 20, 3), 2, [32, 32], [32, 32])
        assert_equal(str(type(model.layers[0])), "<class 'keras.layers.normalization.BatchNormalization'>",
                     'Wrong layer type.')

    def setUp(self):
        np.random.seed(1234)

if __name__ == '__main__':
    unittest.main()
