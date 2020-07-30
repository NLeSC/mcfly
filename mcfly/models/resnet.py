# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution1D, BatchNormalization, ReLU, Add, \
    Input, GlobalAvgPool1D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
from argparse import Namespace
from .base_hyperparameter_generator import generate_base_hyperparameter_set


class ResNet:
    """Generate ResNet model and hyperparameters.
    """
    def __init__(self,
                 x_shape,
                 number_of_classes,
                 metrics = ['accuracy'],
                 **settings):
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
        resnet_min_network_dept : int
            minimum number of Inception modules in ResNet model
        resnet_max_network_dept : int
            maximum number of Inception modules in ResNet model
        resnet_min_filters_number : int
            minimum number of filters per Conv layer in ResNet model
        resnet_max_filters_number : int
            maximum number of filters per Conv layer in ResNet model
        resnet_min_max_kernel_size : int
            minimum size of CNN kernels in ResNet model
        resnet_max_max_kernel_size : int
            maximum size of CNN kernels in ResNet model
        """
        self.model_name = "ResNet"
        self.x_shape = x_shape
        self.number_of_classes = number_of_classes
        self.metrics = metrics

        # Set default parameters
        self.defaults = {
            'resnet_min_network_depth': 2,
            'resnet_max_network_depth': 5,
            'resnet_min_filters_number': 32,
            'resnet_max_filters_number': 128,
            'resnet_min_max_kernel_size': 8,
            'resnet_max_max_kernel_size': 32,
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
        """Generate a hyperparameter set that define a ResNet model.

        Returns
        ----------
        hyperparameters : dict
            parameters for a ResNet model
        """
        params = Namespace(**self.settings)
        hyperparameters = generate_base_hyperparameter_set(params.low_lr,
                                                            params.high_lr,
                                                            params.low_reg,
                                                            params.high_reg)
        hyperparameters['network_depth'] = np.random.randint(params.resnet_min_network_depth,
                                                              params.resnet_max_network_depth + 1)
        hyperparameters['min_filters_number'] = np.random.randint(params.resnet_min_filters_number,
                                                                  params.resnet_max_filters_number + 1)
        hyperparameters['max_kernel_size'] = np.random.randint(params.resnet_min_max_kernel_size,
                                                                params.resnet_max_max_kernel_size + 1)
        return hyperparameters


    def create_model(
            self,
            min_filters_number,
            max_kernel_size,
            network_depth=3,
            learning_rate=0.01,
            regularization_rate=0.01):
        """
        Generate a ResNet model (see also https://arxiv.org/pdf/1611.06455.pdf).

        The compiled Keras model is returned.

        Parameters
        ----------
        min_filters_number : int
            Number of filters for first convolutional layer
        max_kernel_size: int,
            Maximum kernel size for convolutions within Inception module
        network_depth : int
            Depth of network, i.e. number of Inception modules to stack.
            Default is 3.
        learning_rate : float
            Set learning rate. Default is 0.01.
        regularization_rate : float
            Set regularization rate. Default is 0.01.

        Returns
        -------
        model : Keras model
            The compiled Keras model
        """
        dim_length = self.x_shape[1]  # number of samples in a time series
        dim_channels = self.x_shape[2]  # number of channels
        weightinit = 'lecun_uniform'
        regularization = 0

        def conv_bn_relu_3_sandwich(x, filters, kernel_size):
            first_x = x
            for i in range(3):
                x = Convolution1D(filters, kernel_size, padding='same',
                                         kernel_initializer=weightinit,
                                         kernel_regularizer=l2(regularization))(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)

            first_x = Convolution1D(filters, kernel_size=1, padding='same',
                                           kernel_initializer=weightinit,
                                           kernel_regularizer=l2(regularization))(x)
            x = Add()([x, first_x])
            return x

        x = Input((dim_length, dim_channels))
        inputs = x

        x = BatchNormalization()(inputs)  # Added batchnorm (not in original paper)

        # Define/guess filter sizes and kernel sizes
        # Logic here is that kernals become smaller while the number of filters increases
        kernel_sizes = [max(3, int(max_kernel_size // (1.41 ** i))) for i in range(network_depth)]
        filter_numbers = [int(min_filters_number * (1.41 ** i)) for i in range(network_depth)]

        for i in range(network_depth):
            x = conv_bn_relu_3_sandwich(x, filter_numbers[i], kernel_sizes[i])

        x = GlobalAvgPool1D()(x)
        output_layer = Dense(self.number_of_classes, activation='softmax')(x)

        # Create model and compile
        model = Model(inputs=inputs, outputs=output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=learning_rate),
                      metrics=self.metrics)

        return model
