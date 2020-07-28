import numpy as np
from mcfly.models import model_cnn


# Tests for CNN model:
def test_cnn_starts_with_batchnorm():
    """ CNN models should always start with a batch normalization layer. """
    model_type = model_cnn((None, 20, 3), 2)
    model = model_type.create_model({"filters": [32, 32],
                                     "fc_hidden_nodes": 100})
    assert 'BatchNormalization' in str(type(model.layers[0])), 'Wrong layer type.'

def test_cnn_fc_nodes():
    """ CNN model should have number of dense nodes defined by user. """
    fc_hidden_nodes = 101
    model_type = model_cnn((None, 20, 3), 2)
    model = model_type.create_model({"filters": [32, 32],
                                     "fc_hidden_nodes": fc_hidden_nodes})
    dense_layer = [l for l in model.layers if 'Dense' in str(l)][0]
    assert dense_layer.output_shape[1] == fc_hidden_nodes, 'Wrong number of fc nodes.'

# def test_cnn_batchnorm_dim(self):
#     """"The output shape of the batchnorm should be (None, nr_timesteps, nr_filters)"""
#     model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
#     batchnormlay = model.layers[2]
#     assert batchnormlay.output_shape == (None, 20, 32)

# def test_cnn_enough_batchnorm(self):
#     """CNN model should contain as many batch norm layers as it has activations layers"""
#     model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100)
#     batch_norm_layers = len([l for l in model.layers if 'BatchNormalization' in str(l)])
#     activation_layers = len([l for l in model.layers if 'Activation' in str(l)])
#     assert batch_norm_layers == activation_layers

# def test_cnn_metrics(self):
#     """CNN model should be compiled with the metrics that we give it"""
#     metrics = ['accuracy', 'mae']
#     model = modelgen.generate_CNN_model((None, 20, 3), 2, [32, 32], 100, metrics=metrics)
#     model_metrics = [m.name for m in model.metrics]
#     assert model_metrics == metrics or model_metrics == ['acc', 'mean_absolute_error']

# def test_CNN_hyperparameters_nrlayers(self):
#     """ Number of Conv layers from range [4, 4] should be 4. """
#     custom_settings = self.get_default()
#     kwargs = {'cnn_min_layers' : 4,
#               'cnn_max_layers' : 4}
#     # Replace default parameters with input
#     for key, value in kwargs.items():
#         if key in custom_settings:
#             custom_settings[key] = value
#     hyperparams = modelgen.generate_CNN_hyperparameter_set(custom_settings)
#     assert len(hyperparams.get('filters')) == 4

# def test_CNN_hyperparameters_fcnodes(self):
#     """ Number of fc nodes from range [123, 123] should be 123. """
#     custom_settings = self.get_default()
#     kwargs = {'cnn_min_fc_nodes' : 123,
#               'cnn_max_fc_nodes' : 123}
#     # Replace default parameters with input
#     for key, value in kwargs.items():
#         if key in custom_settings:
#             custom_settings[key] = value
#     hyperparams = modelgen.generate_CNN_hyperparameter_set(custom_settings)
#     assert hyperparams.get('fc_hidden_nodes') == 123

