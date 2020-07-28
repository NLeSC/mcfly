from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Convolution1D, Lambda, \
    Convolution2D, Flatten,\
    Reshape, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
from argparse import Namespace
from .models.base_hyperparameter_generator import generate_base_hyperparameter_set


class model_cnn:
    """Generate CNN model and hyperparameters.
    """
    def __init__(self, x_shape, number_of_classes, **settings):
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
        cnn_min_layers : int
            minimum of Conv layers in CNN model
        cnn_max_layers : int
            maximum of Conv layers in CNN model
        cnn_min_filters : int
            minimum number of filters per Conv layer in CNN model
        cnn_max_filters : int
            maximum number of filters per Conv layer in CNN model
        cnn_min_fc_nodes : int
            minimum number of hidden nodes per Dense layer in CNN model
        cnn_max_fc_nodes : int
            maximum number of hidden nodes per Dense layer in CNN model
        """
        self.model_name = "CNN"
        self.x_shape = x_shape
        self.number_of_classes = number_of_classes

        # Set default parameters
        self.defaults = {
            'metrics': ['accuracy'],
            'cnn_min_layers': 1,
            'cnn_max_layers': 10,
            'cnn_min_filters': 10,
            'cnn_max_filters': 100,
            'cnn_min_fc_nodes': 10,
            'cnn_max_fc_nodes': 2000
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
        """Generate a hyperparameter set that define a CNN model.

        Returns
        ----------
        hyperparameters : dict
            parameters for a CNN model
        """
        params = Namespace(**self.settings)
        hyperparameters = generate_base_hyperparameter_set(params.low_lr,
                                                            params.high_lr,
                                                            params.low_reg,
                                                            params.high_reg)
        number_of_layers = np.random.randint(params.cnn_min_layers,
                                             params.cnn_max_layers + 1)
        hyperparameters['filters'] = np.random.randint(params.cnn_min_filters,
                                                       params.cnn_max_filters + 1,
                                                       number_of_layers)
        hyperparameters['fc_hidden_nodes'] = np.random.randint(params.cnn_min_fc_nodes,
                                                               params.cnn_max_fc_nodes + 1)
        return hyperparameters

    def create_model(self, filters, fc_hidden_nodes,
                     learning_rate=0.01, regularization_rate=0.01):
        """
        Generate a convolutional neural network (CNN) model.

        The compiled Keras model is returned.

        Parameters
        ----------
        filters : list of ints
            number of filters for each convolutional layer
        fc_hidden_nodes : int
            number of hidden nodes for the hidden dense layer
        learning_rate : float
            learning rate
        regularization_rate : float
            regularization rate

        Returns
        -------
        model : Keras model
            The compiled Keras model
        """
        dim_length = self.x_shape[1]  # number of samples in a time series
        dim_channels = self.x_shape[2]  # number of channels
        outputdim = self.number_of_classes
        weightinit = 'lecun_uniform'  # weight initialization
        model = Sequential()
        model.add(
            BatchNormalization(
                input_shape=(
                    dim_length,
                    dim_channels)))
        for filter_number in filters:
            model.add(Convolution1D(filter_number, kernel_size=3, padding='same',
                                    kernel_regularizer=l2(regularization_rate),
                                    kernel_initializer=weightinit))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(units=fc_hidden_nodes,
                        kernel_regularizer=l2(regularization_rate),
                        kernel_initializer=weightinit))  # Fully connected layer
        model.add(Activation('relu'))  # Relu activation
        model.add(Dense(units=outputdim, kernel_initializer=weightinit))
        model.add(BatchNormalization())
        model.add(Activation("softmax"))  # Final classification layer

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=learning_rate),
                      metrics=self.metrics)

        return model
