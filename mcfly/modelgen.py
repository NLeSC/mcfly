#
# mcfly
#
# Copyright 2017 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Lambda, \
    Convolution2D, Flatten, \
    Reshape, LSTM, Dropout, TimeDistributed, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np


def generate_models(
        x_shape, number_of_classes, number_of_models=5, metrics=['accuracy'],
        model_type=None,
        cnn_min_layers=1, cnn_max_layers=10,
        cnn_min_filters=10, cnn_max_filters=100,
        cnn_min_fc_nodes=10, cnn_max_fc_nodes=2000,
        deepconvlstm_min_conv_layers=1, deepconvlstm_max_conv_layers=10,
        deepconvlstm_min_conv_filters=10, deepconvlstm_max_conv_filters=100,
        deepconvlstm_min_lstm_layers=1, deepconvlstm_max_lstm_layers=5,
        deepconvlstm_min_lstm_dims=10, deepconvlstm_max_lstm_dims=100,
        low_lr=1, high_lr=4, low_reg=1, high_reg=4
):
    """
    Generate one or multiple untrained Keras models with random hyperparameters.

    Parameters
    ----------
    x_shape : tuple
        Shape of the input dataset: (num_samples, num_timesteps, num_channels)
    number_of_classes : int
        Number of classes for classification task
    number_of_models : int
        Number of models to generate
    metrics : list
        Metrics to calculate on the validation set.
        See https://keras.io/metrics/ for possible values.
    model_type : str, optional
        Type of model to build: 'CNN' or 'DeepConvLSTM'.
        Default option None generates both models.
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
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`

    Returns
    -------
    models : list
        List of compiled models
    """
    models = []
    for _ in range(0, number_of_models):
        if model_type is None:  # random model choice:
            current_model_type = 'CNN' if np.random.random(
            ) < 0.5 else 'DeepConvLSTM'
        else:  # user-defined model choice:
            current_model_type = model_type
        generate_model = None
        if current_model_type == 'CNN':
            generate_model = generate_CNN_model  # generate_model is a function
            hyperparameters = generate_CNN_hyperparameter_set(
                min_layers=cnn_min_layers, max_layers=cnn_max_layers,
                min_filters=cnn_min_filters, max_filters=cnn_max_filters,
                min_fc_nodes=cnn_min_fc_nodes, max_fc_nodes=cnn_max_fc_nodes,
                low_lr=low_lr, high_lr=high_lr, low_reg=low_reg,
                high_reg=high_reg)
        if current_model_type == 'DeepConvLSTM':
            generate_model = generate_DeepConvLSTM_model
            hyperparameters = generate_DeepConvLSTM_hyperparameter_set(
                min_conv_layers=deepconvlstm_min_conv_layers,
                max_conv_layers=deepconvlstm_max_conv_layers,
                min_conv_filters=deepconvlstm_min_conv_filters,
                max_conv_filters=deepconvlstm_max_conv_filters,
                min_lstm_layers=deepconvlstm_min_lstm_layers,
                max_lstm_layers=deepconvlstm_max_lstm_layers,
                min_lstm_dims=deepconvlstm_min_lstm_dims,
                max_lstm_dims=deepconvlstm_max_lstm_dims,
                low_lr=low_lr, high_lr=high_lr, low_reg=low_reg,
                high_reg=high_reg)
        models.append(
            (generate_model(x_shape, number_of_classes, metrics=metrics, **hyperparameters),
             hyperparameters, current_model_type))
    return models


def generate_DeepConvLSTM_model(
        x_shape, class_number, filters, lstm_dims, learning_rate=0.01,
        regularization_rate=0.01, metrics=['accuracy']):
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

    Parameters
    ----------
    x_shape : tuple
        Shape of the input dataset: (num_samples, num_timesteps, num_channels)
    class_number : int
        Number of classes for classification task
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
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    output_dim = class_number  # number of classes
    weightinit = 'lecun_uniform'  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
    # reshape a 2 dimensional array per file/person/object into a
    # 3 dimensional array
    model.add(
        Reshape(target_shape=(dim_length, dim_channels, 1)))
    for filt in filters:
        # filt: number of filters used in a layer
        # filters: vector of filt values
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
                  metrics=metrics)

    return model


def generate_CNN_model(x_shape, class_number, filters, fc_hidden_nodes,
                       learning_rate=0.01, regularization_rate=0.01,
                       metrics=['accuracy']):
    """
    Generate a convolutional neural network (CNN) model.

    The compiled Keras model is returned.

    Parameters
    ----------
    x_shape : tuple
        Shape of the input dataset: (num_samples, num_timesteps, num_channels)
    class_number : int
        Number of classes for classification task
    filters : list of ints
        number of filters for each convolutional layer
    fc_hidden_nodes : int
        number of hidden nodes for the hidden dense layer
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
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    outputdim = class_number  # number of classes
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
                  metrics=metrics)

    return model


def generate_CNN_hyperparameter_set(min_layers=1, max_layers=10,
                                    min_filters=10, max_filters=100,
                                    min_fc_nodes=10, max_fc_nodes=2000,
                                    low_lr=1, high_lr=4, low_reg=1,
                                    high_reg=4):
    """ Generate a hyperparameter set that define a CNN model.

    Parameters
    ----------
    min_layers : int
        minimum of Conv layers
    max_layers : int
        maximum of Conv layers
    min_filters : int
        minimum number of filters per Conv layer
    max_filters : int
        maximum number of filters per Conv layer
    min_fc_nodes : int
        minimum number of hidden nodes per Dense layer
    max_fc_nodes : int
        maximum number of hidden nodes per Dense layer
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`

    Returns
    ----------
    hyperparameters : dict
        parameters for a CNN model
    """
    hyperparameters = generate_base_hyper_parameter_set(
        low_lr, high_lr, low_reg, high_reg)
    number_of_layers = np.random.randint(min_layers, max_layers + 1)
    hyperparameters['filters'] = np.random.randint(
        min_filters, max_filters + 1, number_of_layers)
    hyperparameters['fc_hidden_nodes'] = np.random.randint(
        min_fc_nodes, max_fc_nodes + 1)
    return hyperparameters


def generate_DeepConvLSTM_hyperparameter_set(
        min_conv_layers=1, max_conv_layers=10,
        min_conv_filters=10, max_conv_filters=100,
        min_lstm_layers=1, max_lstm_layers=5,
        min_lstm_dims=10, max_lstm_dims=100,
        low_lr=1, high_lr=4, low_reg=1, high_reg=4):
    """ Generate a hyperparameter set that defines a DeepConvLSTM model.

    Parameters
    ----------
    min_conv_layers : int
        minimum number of Conv layers in DeepConvLSTM model
    max_conv_layers : int
        maximum number of Conv layers in DeepConvLSTM model
    min_conv_filters : int
        minimum number of filters per Conv layer in DeepConvLSTM model
    max_conv_filters : int
        maximum number of filters per Conv layer in DeepConvLSTM model
    min_lstm_layers : int
        minimum number of Conv layers in DeepConvLSTM model
    max_lstm_layers : int
        maximum number of Conv layers in DeepConvLSTM model
    min_lstm_dims : int
        minimum number of hidden nodes per LSTM layer in DeepConvLSTM model
    max_lstm_dims : int
        maximum number of hidden nodes per LSTM layer in DeepConvLSTM model
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`

    Returns
    ----------
    hyperparameters: dict
        hyperparameters for a DeepConvLSTM model
    """
    hyperparameters = generate_base_hyper_parameter_set(
        low_lr, high_lr, low_reg, high_reg)
    number_of_conv_layers = np.random.randint(
        min_conv_layers, max_conv_layers + 1)
    hyperparameters['filters'] = np.random.randint(
        min_conv_filters, max_conv_filters + 1, number_of_conv_layers).tolist()
    number_of_lstm_layers = np.random.randint(
        min_lstm_layers, max_lstm_layers + 1)
    hyperparameters['lstm_dims'] = np.random.randint(
        min_lstm_dims, max_lstm_dims + 1, number_of_lstm_layers).tolist()
    return hyperparameters


def generate_base_hyper_parameter_set(
        low_lr=1,
        high_lr=4,
        low_reg=1,
        high_reg=4):
    """ Generate a base set of hyperparameters that are necessary for any
    model, but sufficient for none.

    Parameters
    ----------
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`

    Returns
    -------
    hyperparameters : dict
        basis hyperpameters
    """
    hyperparameters = {}
    hyperparameters['learning_rate'] = get_learning_rate(low_lr, high_lr)
    hyperparameters['regularization_rate'] = get_regularization(
        low_reg, high_reg)
    return hyperparameters


def get_learning_rate(low=1, high=4):
    """ Return random learning rate 10^-n where n is sampled uniformly between
    low and high bounds.

    Parameters
    ----------
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    learning_rate : float
        learning rate
    """
    result = 10 ** (-np.random.uniform(low, high))
    return result


def get_regularization(low=1, high=4):
    """ Return random regularization rate 10^-n where n is sampled uniformly
    between low and high bounds.

    Parameters
    ----------
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    regularization_rate : float
        regularization rate
    """
    return 10 ** (-np.random.uniform(low, high))
