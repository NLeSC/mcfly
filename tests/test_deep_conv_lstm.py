# -*- coding: utf-8 -*-
import unittest
from mcfly.models import DeepConvLSTM
from test_modelgen import get_default

class DeepConvLSTMSuite(unittest.TestCase):
    """
    Tests cases for DeepconvLSTM models.
    """

    def test_deepconvlstm_batchnorm_dim(self):
        """The output shape of the batchnorm should be (None, nr_timesteps, nr_channels, nr_filters)"""
        model_type = DeepConvLSTM((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "lstm_dims": [32, 32]})

        batchnormlay = model.layers[3]
        assert batchnormlay.output.shape == (None, 20, 3, 32)

    def test_deepconvlstm_enough_batchnorm(self):
        """LSTM model should contain as many batch norm layers as it has activations layers"""
        model_type = DeepConvLSTM((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32, 32],
                                           "lstm_dims": [32, 32, 32]})

        batch_norm_layers = len([layer for layer in model.layers if 'BatchNormalization' in str(layer)])
        activation_layers = len([layer for layer in model.layers if 'Activation' in str(layer)])
        assert batch_norm_layers == activation_layers

    def test_DeepConvLSTM_hyperparameters_nrconvlayers(self):
        """ Number of Conv layers from range [4, 4] should be 4. """
        custom_settings = get_default()
        kwargs = {'deepconvlstm_min_conv_layers': 4,
                  'deepconvlstm_max_conv_layers': 4}
        # Replace default parameters with input
        for key, value in kwargs.items():
            if key in custom_settings:
                custom_settings[key] = value

        model_type = DeepConvLSTM(None, None, **custom_settings)
        hyperparams = model_type.generate_hyperparameters()

        assert len(hyperparams.get('filters')) == 4

    def test_deepconvlstm_starts_with_batchnorm(self):
        """ DeepConvLSTM models should always start with a batch normalization layer. """
        model_type = DeepConvLSTM((None, 20, 3), 2)
        model = model_type.create_model(**{"filters": [32, 32],
                                           "lstm_dims": [32, 32]})

        assert 'BatchNormalization' in str(type(model.layers[0])), 'Wrong layer type.'


if __name__ == '__main__':
    unittest.main()
