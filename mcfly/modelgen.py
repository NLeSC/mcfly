#
# mcfly
#
# Copyright 2020 Netherlands eScience Center
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

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Convolution1D, Lambda, \
    Convolution2D, Flatten, Input,\
    Reshape, LSTM, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
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
        IT_min_network_depth=3, IT_max_network_depth=6,
        IT_min_filters_number=32, IT_max_filters_number=96,
        IT_min_max_kernel_size=10, IT_max_max_kernel_size=100,
        low_lr=1, high_lr=4, low_reg=1, high_reg=4 # TODO: use centralized default parameter file (e.g. yaml)
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
    
    # Limit parameter space based on input
    # -------------------------------------------------------------------------
    if IT_max_max_kernel_size > x_shape[1]:
        print("Set maximum kernel size for InceptionTime models to number of timesteps.")
        IT_max_max_kernel_size = x_shape[1]
    
    model_types = ['CNN', 'DeepConvLSTM', 'InceptionTime']
    model_types_selected = []
    for i in range(int(np.ceil(number_of_models/len(model_types)))):
        np.random.shuffle(model_types)
        model_types_selected.extend(model_types)

    # Create list of Keras models
    # -------------------------------------------------------------------------    
    models = []
    for i in range(0, number_of_models):
        if model_type is None:  # random model choice:
            current_model_type = model_types_selected[i]
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
        if current_model_type == 'InceptionTime':
            generate_model = generate_InceptionTime_model
            hyperparameters = generate_InceptionTime_hyperparameter_set(
                min_network_depth=IT_min_network_depth, 
                max_network_depth=IT_max_network_depth,
                min_filters_number=IT_min_filters_number, 
                max_filters_number=IT_max_filters_number,
                min_max_kernel_size=IT_min_max_kernel_size, 
                max_max_kernel_size=IT_max_max_kernel_size,
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


def generate_InceptionTime_model(input_shape, 
                                 class_number, 
                                 filters_number, 
                                 network_depth=6,
                                 use_residual=True, 
                                 use_bottleneck=True, 
                                 max_kernel_size = 20,
                                 learning_rate=0.01, 
                                 regularization_rate=0.01,
                                 metrics=['accuracy']):
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
    dim_length = input_shape[1]  # number of samples in a time series
    dim_channels = input_shape[2]  # number of channels
    outputdim = class_number  # number of classes
    weightinit = 'lecun_uniform'  # weight initialization
    bottleneck_size = 32
    
    # TODO: switch to Sequential() keras syntax ?
    #model = Sequential()
    #model.add(Input((dim_length, dim_channels)))
    
    def inception_module(input_tensor, stride=1, activation='linear'):

        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                  padding='same', 
                                                  activation=activation, 
                                                  kernel_initializer=weightinit,
                                                  use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(layers.Conv1D(filters=filters_number, 
                                           kernel_size=kernel_size,
                                           strides=stride, 
                                           padding='same', 
                                           activation=activation, 
                                           kernel_initializer=weightinit, 
                                           use_bias=False)(input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = layers.Conv1D(filters=filters_number, 
                               kernel_size=1,
                               padding='same', 
                               activation=activation, 
                               kernel_initializer=weightinit,
                               use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]), 
                                   kernel_size=1,
                                   padding='same', 
                                   kernel_initializer=weightinit, 
                                   use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x
    
    # Build the actual model:
    input_layer = layers.Input((dim_length, dim_channels))
    x = input_layer
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = layers.GlobalAveragePooling1D()(x)
    
    # Final classification layer
    output_layer = layers.Dense(class_number, activation='softmax')(gap_layer)
    
    # Create model and compile
    model = Model(inputs=input_layer, outputs=output_layer)
    
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


def generate_InceptionTime_hyperparameter_set(min_network_depth=3, max_network_depth=6,
                                              min_filters_number=32, max_filters_number=96,
                                              min_max_kernel_size=10, max_max_kernel_size=80,
                                              low_lr=1, high_lr=4, 
                                              low_reg=1, high_reg=4):
    """ Generate a hyperparameter set that define a CNN model.

    Parameters
    ----------
    min_network_dept : int
        minimum number of Inception modules
    max_network_dept : int
        maximum number of Inception modules
    min_filters_number : int
        minimum number of filters per Conv layer
    max_filters_number : int
        maximum number of filters per Conv layer
    min_max_kernel_size : int
        minimum size of CNN kernels
    max_max_kernel_size : int
        maximum size of CNN kernels
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
    hyperparameters['network_depth'] = np.random.randint(min_network_depth, max_network_depth + 1)
    hyperparameters['filters_number'] = np.random.randint(min_filters_number, max_filters_number + 1)
    hyperparameters['max_kernel_size'] = np.random.randint(min_max_kernel_size, max_max_kernel_size + 1)
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
