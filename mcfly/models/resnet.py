# -*- coding: utf-8 -*-
from keras.layers import Activation
from keras.models import Model
from keras.layers import Convolution1D, BatchNormalization, ReLU, Add, \
    Input, GlobalAvgPool1D, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
from argparse import Namespace
from .base_hyperparameter_generator import generate_base_hyperparameter_set
from ..task import Task


class ResNet:
    """Generate ResNet model and hyperparameters.
    """

    model_name = "ResNet"

    def __init__(self,
                 x_shape,
                 number_of_classes,
                 metrics=['accuracy'],
                 resnet_min_network_depth=2,
                 resnet_max_network_depth=5,
                 resnet_min_filters_number=32,
                 resnet_max_filters_number=128,
                 resnet_min_max_kernel_size=8,
                 resnet_max_max_kernel_size=32,
                 **_other):
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
        self.x_shape = x_shape
        self.number_of_classes = number_of_classes
        self.metrics = metrics

        # Set default parameters
        self.settings = {
            'resnet_min_network_depth': resnet_min_network_depth,
            'resnet_max_network_depth': resnet_max_network_depth,
            'resnet_min_filters_number': resnet_min_filters_number,
            'resnet_max_filters_number': resnet_max_filters_number,
            'resnet_min_max_kernel_size': resnet_min_max_kernel_size,
            'resnet_max_max_kernel_size': resnet_max_max_kernel_size,
        }

        # Add missing parameters from default
        for key, value in _other.items():
            if key not in self.settings:
                self.settings[key] = value

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
            regularization_rate=0.01,
            task=Task.classification):
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
        task: str
            Task type, either 'classification' or 'regression'

        Returns
        -------
        model : Keras model
            The compiled Keras model
        """
        dim_length = self.x_shape[1]  # number of samples in a time series
        dim_channels = self.x_shape[2]  # number of channels
        weightinit = 'lecun_uniform'
        regularization = 0  # ignore input on purpose

        def conv_bn_relu_3_sandwich(x, filters, kernel_size):
            first_x = x
            for _ in range(3):
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
        output_layer = Dense(self.number_of_classes)(x)

        if task is Task.classification:
            loss_function = 'categorical_crossentropy'
            output_layer = Activation('softmax')(output_layer)
        elif task is Task.regression:
            loss_function = 'mean_squared_error'

        # Create model and compile
        model = Model(inputs=inputs, outputs=output_layer)

        model.compile(loss=loss_function,
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=self.metrics)

        return model
