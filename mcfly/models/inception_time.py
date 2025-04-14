# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv1D, Concatenate, BatchNormalization, \
    Activation, Add, Input, GlobalAveragePooling1D, Dense, MaxPool1D
from keras.optimizers import Adam
import numpy as np
from argparse import Namespace
from .base_hyperparameter_generator import generate_base_hyperparameter_set
from ..task import Task


class InceptionTime:
    model_name = "InceptionTime"

    def __init__(self,
                 x_shape,
                 number_of_classes,
                 metrics=['accuracy'],
                 IT_min_network_depth=3,
                 IT_max_network_depth=6,
                 IT_min_filters_number=32,
                 IT_max_filters_number=96,
                 IT_min_max_kernel_size=10,
                 IT_max_max_kernel_size=100,
                 **_other
                 ):
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
        IT_min_network_dept : int
            minimum number of Inception modules in InceptionTime model
        IT_max_network_dept : int
            maximum number of Inception modules in InceptionTime model
        IT_min_filters_number : int
            minimum number of filters per Conv layer in InceptionTime model
        IT_max_filters_number : int
            maximum number of filters per Conv layer in InceptionTime model
        IT_min_max_kernel_size : int
            minimum size of CNN kernels in InceptionTime model
        IT_max_max_kernel_size : int
            maximum size of CNN kernels in InceptionTime model
        """
        self.x_shape = x_shape
        self.number_of_classes = number_of_classes
        self.metrics = metrics

        # Limit parameter space based on input
        if IT_max_max_kernel_size > self.x_shape[1]:
            print("Set maximum kernel size for InceptionTime models to number of timesteps.")
            IT_max_max_kernel_size = self.x_shape[1]

        self.settings = {
            'IT_min_network_depth': IT_min_network_depth,
            'IT_max_network_depth': IT_max_network_depth,
            'IT_min_filters_number': IT_min_filters_number,
            'IT_max_filters_number': IT_max_filters_number,
            'IT_min_max_kernel_size': IT_min_max_kernel_size,
            'IT_max_max_kernel_size': IT_max_max_kernel_size
        }

        # Add missing parameters from default
        for key, value in _other.items():
            if key not in self.settings:
                self.settings[key] = value

    def generate_hyperparameters(self):
        """Generate a hyperparameter set for an InceptionTime model.

        Returns
        ----------
        hyperparameters : dict
            Hyperparameter ranges for a InceptionTime model
        """
        params = Namespace(**self.settings)
        hyperparameters = generate_base_hyperparameter_set(params.low_lr,
                                                           params.high_lr,
                                                           params.low_reg,
                                                           params.high_reg)
        hyperparameters['network_depth'] = np.random.randint(params.IT_min_network_depth,
                                                             params.IT_max_network_depth + 1)
        hyperparameters['filters_number'] = np.random.randint(params.IT_min_filters_number,
                                                              params.IT_max_filters_number + 1)
        hyperparameters['max_kernel_size'] = np.random.randint(params.IT_min_max_kernel_size,
                                                               params.IT_max_max_kernel_size + 1)
        return hyperparameters

    def create_model(self,
                     filters_number,
                     network_depth=6,
                     use_residual=True,
                     use_bottleneck=True,
                     max_kernel_size=20,
                     learning_rate=0.01,
                     regularization_rate=0.0,
                     task=Task.classification):
        """
        Generate a InceptionTime model. See Fawaz et al. 2019.

        The compiled Keras model is returned.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input dataset: (num_samples, num_timesteps, num_channels)
        class_number : int
            Number of classes for classification task
        filters_number : int
            number of filters for each convolutional layer
        network_depth : int
            Depth of network, i.e. number of Inception modules to stack.
        use_residual: bool
            If =True, then residual connections are used. Default is True.
        use_bottleneck: bool
            If=True, bottleneck layer is used at the entry of Inception modules.
            Default is true.
        max_kernel_size: int,
            Maximum kernel size for convolutions within Inception module.
        learning_rate : float
            learning rate
        regularization_rate: float
            regularization rate
        task: str
            Task type, either 'classification' or 'regression'

        Returns
        -------
        model : Keras model
            The compiled Keras model
        """
        dim_length = self.x_shape[1]  # number of samples in a time series
        dim_channels = self.x_shape[2]  # number of channels
        weightinit = 'lecun_uniform'  # weight initialization
        bottleneck_size = 32

        def inception_module(input_tensor, stride=1, activation='linear'):

            if use_bottleneck and int(input_tensor.shape[-1]) > 1:
                input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                         padding='same',
                                         activation=activation,
                                         kernel_initializer=weightinit,
                                         use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

            kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
            conv_list = []

            for kernel_size in kernel_sizes:
                conv_list.append(Conv1D(filters=filters_number,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        padding='same',
                                        activation=activation,
                                        kernel_initializer=weightinit,
                                        use_bias=False)(input_inception))

            max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_last = Conv1D(filters=filters_number,
                               kernel_size=1,
                               padding='same',
                               activation=activation,
                               kernel_initializer=weightinit,
                               use_bias=False)(max_pool_1)

            conv_list.append(conv_last)

            x = Concatenate(axis=2)(conv_list)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            return x

        def shortcut_layer(input_tensor, out_tensor):
            shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]),
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=weightinit,
                                use_bias=False)(input_tensor)
            shortcut_y = BatchNormalization()(shortcut_y)

            x = Add()([shortcut_y, out_tensor])
            x = Activation('relu')(x)
            return x

        # Build the actual model:
        input_layer = Input((dim_length, dim_channels))
        x = BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
        input_res = x

        for depth in range(network_depth):
            x = inception_module(x)

            if use_residual and depth % 3 == 2:
                x = shortcut_layer(input_res, x)
                input_res = x

        gap_layer = GlobalAveragePooling1D()(x)

        # Final classification layer
        output_layer = Dense(self.number_of_classes)(gap_layer)

        if task is Task.classification:
            loss_function = 'categorical_crossentropy'
            output_layer = Activation('softmax')(output_layer)
        elif task is Task.regression:
            loss_function = 'mean_squared_error'

            # Create model and compile
        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=loss_function,
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=self.metrics)

        return model
