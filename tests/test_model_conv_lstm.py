import numpy as np
from mcfly.models.model_conv_lstm import Model_ConvLSTM


def test_deepconvlstm_batchnorm_dim():
    """The output shape of the batchnorm should be (None, nr_timesteps, nr_channels, nr_filters)"""
    model_type = Model_ConvLSTM((None, 20, 3), 2)
    model = model_type.create_model(**{"filters": [32, 32],
                                       "lstm_dims": [32, 32]})
    batchnormlay = model.layers[3]
    assert batchnormlay.output_shape == (None, 20, 3, 32)


def test_deepconvlstm_enough_batchnorm():
    """LSTM model should contain as many batch norm layers as it has activations layers"""
    model_type = Model_ConvLSTM((None, 20, 3), 2)
    model = model_type.create_model(**{"filters": [32, 32, 32],
                                       "lstm_dims": [32, 32, 32]})

    batch_norm_layers = len([l for l in model.layers if 'BatchNormalization' in str(l)])
    activation_layers = len([l for l in model.layers if 'Activation' in str(l)])
    assert batch_norm_layers == activation_layers


def test_DeepConvLSTM_hyperparameters_nrconvlayers():
    """ Number of Conv layers from range [4, 4] should be 4. """
    custom_settings = self.get_default()
    settings = {'deepconvlstm_min_conv_layers': 4,
                'deepconvlstm_max_conv_layers': 4,
                'low_lr': 1,
                'high_lr': 4,
                'low_reg': 1,
                'high_reg': 4}
    model_type = Model_ConvLSTM((None, 20, 3), 2, **settings)
    hyperparams = modelgen.generate_DeepConvLSTM_hyperparameter_set(custom_settings)
    assert len(hyperparams.get('filters')) == 4, "Expected different filter number"


def test_deepconvlstm_starts_with_batchnorm():
    """ DeepConvLSTM models should always start with a batch normalization layer. """
    model_type = Model_ConvLSTM((None, 20, 3), 2)
    model = model_type.create_model(**{"filters": [32, 32],
                                       "lstm_dims": [32, 32]})
    assert 'BatchNormalization' in str(type(model.layers[0])), 'Wrong layer type.'
