from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Lambda, \
    Convolution2D, TimeDistributed, \
    Reshape, LSTM, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
from argparse import Namespace
from .base_hyperparameter_generator import generate_base_hyperparameter_set


class Model_ConvLSTM:
    """Generate CNN model and hyperparameters.
    """
    def __init__(self, x_shape, number_of_classes, metrics=['accuracy'], **settings):
        """

        Parameters
        ----------
        x_shape : tuple
            Shape of the input dataset: (num_samples, num_timesteps, num_channels)
        number_of_classes : int
            Number of classes for classification task
        metrics : list
            Metrics to calculate on the validation set.
            See https://keras.io/metrics/ for possible values.
        deepconvlstm_min_conv_layers : int
            minimum number of Conv layers in DeepConvLSTM model
        deepconvlstm_max_conv_layers : int
            maximum number of Conv layers in DeepConvLSTM model
        deepconvlstm_min_conv_filters : int
            minimum number of filters per Conv layer in DeepConvLSTM model
        deepconvlstm_max_conv_filters : int
            maximum number of filters per Conv layer in DeepConvLSTM model
        deepconvlstm_min_lstm_layers : int
            minimum number of Conv layers in DeepConvLSTM model
        deepconvlstm_max_lstm_layers : int
            maximum number of Conv layers in DeepConvLSTM model
        deepconvlstm_min_lstm_dims : int
            minimum number of hidden nodes per LSTM layer in DeepConvLSTM model
        deepconvlstm_max_lstm_dims : int
            maximum number of hidden nodes per LSTM layer in DeepConvLSTM model
        """
        self.model_name = "DeepConvLSTM"
        self.x_shape = x_shape
        self.number_of_classes = number_of_classes
        self.metrics = metrics

        # Set default parameters
        self.defaults = {
            'deepconvlstm_min_conv_layers': 1,
            'deepconvlstm_max_conv_layers': 10,
            'deepconvlstm_min_conv_filters': 10,
            'deepconvlstm_max_conv_filters': 100,
            'deepconvlstm_min_lstm_layers': 1,
            'deepconvlstm_max_lstm_layers': 5,
            'deepconvlstm_min_lstm_dims': 10,
            'deepconvlstm_max_lstm_dims': 100,
            }

        # Replace default parameters with input
        for key, value in settings.items():
            if key in self.defaults:
                print("The value of {} is set from {} (default) to {}".format(key, self.defaults[key], value))

        # Add missing parameters from default
        for key, value in self.defaults.items():
            if key not in settings:
                settings[key] = value
        self.settings = settings

    def generate_hyperparameters(self):
        """Generate a hyperparameter set that defines a DeepConvLSTM model.

        Returns
        ----------
        hyperparameters : dict
            parameters for a CNN model
        """
        params = Namespace(**self.settings)
        hyperparameters = generate_base_hyper_parameter_set(params.low_lr,
                                                            params.high_lr,
                                                            params.low_reg,
                                                            params.high_reg)
        number_of_conv_layers = np.random.randint(params.deepconvlstm_min_conv_layers,
                                                  params.deepconvlstm_max_conv_layers + 1)
        hyperparameters['filters'] = np.random.randint(params.deepconvlstm_min_conv_filters,
                                                       params.deepconvlstm_max_conv_filters + 1,
                                                       number_of_conv_layers).tolist()
        number_of_lstm_layers = np.random.randint(params.deepconvlstm_min_lstm_layers,
                                                  params.deepconvlstm_max_lstm_layers + 1)
        hyperparameters['lstm_dims'] = np.random.randint(params.deepconvlstm_min_lstm_dims,
                                                         params.deepconvlstm_max_lstm_dims + 1,
                                                         number_of_lstm_layers).tolist()
        return hyperparameters

    def create_model(self, filters, lstm_dims, learning_rate=0.01,
                     regularization_rate=0.01):
        """Generate a model with convolution and LSTM layers.

        See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

        Parameters
        ----------
        filters : list of ints
            number of filters for each convolutional layer
        lstm_dims : list of ints
            number of hidden nodes for each LSTM layer
        learning_rate : float
            learning rate
        regularization_rate : float
            regularization rate
        metrics : list
            Metrics to calculate on the validation set.
            See https://keras.io/metrics/ for possible values.

        Returns
        -------
        model : Keras model
            The compiled Keras model
        """
        dim_length = self.x_shape[1]  # number of samples in a time series
        dim_channels = self.x_shape[2]  # number of channels
        weightinit = 'lecun_uniform'  # weight initialization
        model = Sequential()  # initialize model
        model.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
        # reshape a 2 dimensional array per file/person/object into a
        # 3 dimensional array
        model.add(
            Reshape(target_shape=(dim_length, dim_channels, 1)))
        for filt in filters:
            # filt: number of filters used in a layer
            model.add(
                Convolution2D(filt, kernel_size=(3, 1), padding='same',
                              kernel_regularizer=l2(regularization_rate),
                              kernel_initializer=weightinit))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        # reshape 3 dimensional array back into a 2 dimensional array,
        # but now with more dept as we have the the filters for each channel
        model.add(Reshape(target_shape=(dim_length, filters[-1] * dim_channels)))

        for lstm_dim in lstm_dims:
            model.add(LSTM(units=lstm_dim, return_sequences=True,
                           activation='tanh'))

        model.add(Dropout(0.5))  # dropout before the dense layer
        # set up final dense layer such that every timestamp is given one
        # classification
        model.add(
            TimeDistributed(
                Dense(units=output_dim, kernel_regularizer=l2(regularization_rate))))
        model.add(Activation("softmax"))
        # Final classification layer - per timestep
        model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=learning_rate),
                      metrics=self.metrics)

        return model
