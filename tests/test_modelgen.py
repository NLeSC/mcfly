import numpy as np
import unittest
import pytest
from tensorflow.keras.models import Model
from mcfly import modelgen
from mcfly.models import CNN, ConvLSTM, ResNet, InceptionTime


# TODO: Move this to an utils file, or obtain it from other source?
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

class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def _generate_train_data(self, x_shape, nr_classes):
        X_train = np.random.rand(1, *x_shape[1:])
        y_train = np.random.randint(0, 1, size=(1, nr_classes))
        return X_train, y_train



    # Tests for ResNet model:
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

        add_layers = [str(type(l)) for l in model.layers if 'Add' in str(type(l))]
        assert len(add_layers) == depths, 'Wrong number of residual modules (network depths).'


    def test_ResNet_first_module_dim(self):
        """"The output shape throughout the first residual module should be (None, nr_timesteps, min_filters_number)"""
        min_filters_number = 16

        model_type = ResNet((None, 30, 5), 2)
        model = model_type.create_model(min_filters_number, 20)

        firstConvlayer = model.layers[2]
        firstAddlayer = model.layers[12]
        assert firstConvlayer.output_shape == (None, 30, min_filters_number)
        assert firstAddlayer.output_shape == (None, 30, min_filters_number)

    def test_ResNet_metrics(self):
        """ResNet model should be compiled with the metrics that we give it"""
        metrics = ['accuracy', 'mae']
        x_shape = (None, 20, 3)
        nr_classes = 2
        X_train, y_train = self._generate_train_data(x_shape, nr_classes)

        model_type = ResNet(x_shape, nr_classes, metrics=metrics)
        model = model_type.create_model(16, 20)
        model.fit(X_train, y_train, epochs=1)

        model_metrics = [m.name for m in model.metrics]
        for metric in metrics:
            assert metric in model_metrics

    def test_ResNet_hyperparameters(self):
        """ Network depth from range [4,4] should be 4.
        Maximum kernal size from range [10, 10] should be 10.
        Minimum filter number from range [16, 16] should be 16.  """
        custom_settings = get_default()
        kwargs = {'resnet_min_network_depth': 4,
                  'resnet_mmax_network_depth': 4,
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
        """ ResNet model should have depth (number of residual modules) as defined by user. """
        depths = 3

        model_type = InceptionTime((None, 20, 3), 2)
        model = model_type.create_model(16, network_depth=depths)

        concat_layers = [str(type(l)) for l in model.layers if 'concatenate' in str(type(l)).lower()]
        assert len(concat_layers) == depths, 'Wrong number of inception modules (network depths).'

    def test_InceptionTime_first_module_dim(self):
        """"The output shape throughout the first residual module should be (None, nr_timesteps, min_filters_number)"""
        min_filters_number = 16

        model_type = InceptionTime((None, 30, 5), 2)
        model = model_type.create_model(min_filters_number)

        secondConvlayer = model.layers[5]
        firstConcatlayer = model.layers[8]
        assert secondConvlayer.output_shape == (None, 30, min_filters_number)
        assert firstConcatlayer.output_shape == (None, 30, min_filters_number * 4)

    def test_InceptionTime_metrics(self):
        """ResNet model should be compiled with the metrics that we give it"""
        metrics = ['accuracy', 'mae']
        x_shape = (None, 20, 3)
        nr_classes = 2
        X_train, y_train = self._generate_train_data(x_shape, nr_classes)

        model_type = InceptionTime(x_shape, nr_classes, metrics=metrics)
        model = model_type.create_model(16)
        model.fit(X_train, y_train)

        model_metrics = [m.name for m in model.metrics]
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

    # Tests for general mcfly functionality:
    def test_generate_models_metrics(self):
        """ Test if correct number of models is generated and if metrics is correct. """
        x_shape = (None, 20, 10)
        nr_classes = 2
        X_train, y_train = self._generate_train_data(x_shape, nr_classes)
        n_models = 5

        models = modelgen.generate_models(x_shape, nr_classes, n_models)
        for model in models:
            model[0].fit(X_train, y_train, epochs = 1)

        model, hyperparams, modeltype = models[0]
        model_metrics = [m.name for m in model.metrics]
        assert "accuracy" in model_metrics, "Not found accuracy for model {}. Found {}".format(
            modeltype, model_metrics)
        assert len(models) == n_models, "Expecting {} models, found {} models".format(
            n_models, len(models))

    def test_generate_models_pass_model_object(self):
        """ Test if model class can be passed as model_types input."""
        x_shape = (None, 20, 10)
        nr_classes = 2
        X_train, y_train = self._generate_train_data(x_shape, nr_classes)
        n_models = 4

        models = modelgen.generate_models(x_shape, nr_classes, n_models,
                                          model_types=['CNN', ResNet])
        created_model_names = list(set([x[2] for x in models]))
        created_model_names.sort()
        assert len(models) == 4, "Expected number of models to be 4"
        assert created_model_names == ["CNN", "ResNet"], "Expected different model names."
        for model in models:
            assert isinstance(model[0], Model), "Expected keras model."

    def test_generate_models_exception(self):
        """ Test expected generate_models exception."""
        x_shape = (None, 20, 10)
        nr_classes = 2
        X_train, y_train = self._generate_train_data(x_shape, nr_classes)
        n_models = 2

        with pytest.raises(NameError, match="Unknown model name, 'wrong_entry'."):
            models = modelgen.generate_models(x_shape, nr_classes, n_models,
                                              model_types=['CNN', "wrong_entry"])

    def setUp(self):
        np.random.seed(1234)


if __name__ == '__main__':
    unittest.main()
