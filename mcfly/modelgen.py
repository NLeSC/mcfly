from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Flatten, MaxPooling1D, Lambda, Convolution2D, Flatten, Reshape, LSTM, Dropout, TimeDistributed, Permute, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np


def generate_models(x_shape, number_of_classes, number_of_models=5, model_type=None, **kwargs):
    """ Generate one or multiple Keras models with random (default), or predefined, hyperparameters."""
    models = []
    for _ in range(0, number_of_models):
        if model_type is None:  # random model choice:
            current_model_type = 'CNN' if np.random.random(
            ) < 0.5 else 'DeepConvLSTM'
        else:  # user-defined model choice:
            current_model_type = model_type

        if current_model_type == 'CNN':
            generate_model = generate_CNN_model  # object is a function
            generate_hyperparameter_set = generate_CNN_hyperparameter_set
        if current_model_type == 'DeepConvLSTM':
            generate_model = generate_DeepConvLSTM_model  # object is a function
            generate_hyperparameter_set = generate_DeepConvLSTM_hyperparameter_set
        hyperparameters = generate_hyperparameter_set(**kwargs)
        models.append(
            (generate_model(x_shape, number_of_classes, **hyperparameters), hyperparameters, current_model_type))
    return models


def generate_DeepConvLSTM_model(x_shape, class_number, filters, lstm_dims, learning_rate=0.01,
                                regularization_rate=0.01):
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

    The compiled Keras model is returned.
    """
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    output_dim = class_number  # number of classes
    weightinit = 'lecun_uniform'  # weight initialization
    model = Sequential()  # initialize model
    # reshape a 2 dimensional array per file/person/object into a
    # 3 dimensional array
    model.add(
        Reshape(target_shape=(1, dim_length, dim_channels), input_shape=(dim_length, dim_channels)))
    for filt in filters:
        # filt: number of filters used in a layer
        # filters: vector of filt values
        model.add(
            Convolution2D(filt, nb_row=3, nb_col=1, border_mode='same',
                          W_regularizer=l2(regularization_rate), init=weightinit))
        model.add(Activation('relu'))
    # reshape 3 dimensional array back into a 2 dimensional array,
    # but now with more dept as we have the the filters for each channel
    model.add(Reshape(target_shape=(dim_length, filters[-1] * dim_channels)))

    for lstm_dim in lstm_dims:
        model.add(LSTM(output_dim=lstm_dim, return_sequences=True,
                       activation='tanh'))

    model.add(Dropout(0.5))  # dropout before the dense layer
    # set up final dense layer such that every timestamp is given one classification
    model.add(TimeDistributed(Dense(output_dim, W_regularizer=l2(regularization_rate))))
    model.add(Activation("softmax"))
    # Final classification layer - per timestep
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


def generate_CNN_model(x_shape, class_number, filters, fc_hidden_nodes,
                       learning_rate=0.01, regularization_rate=0.01):
    """
    Generate a convolutional neural network (CNN) model.

    The compiled Keras model is returned.
    """
    dim_length = x_shape[1]  # number of samples in a time series
    dim_channels = x_shape[2]  # number of channels
    outputdim = class_number  # number of classes
    weightinit = 'lecun_uniform'  # weight initialization
    model = Sequential()
    model.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
    for filter_number in filters:
        model.add(Convolution1D(filter_number, 3, border_mode='same',
                                W_regularizer=l2(regularization_rate), init=weightinit))
        model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(output_dim=fc_hidden_nodes,
                    W_regularizer=l2(regularization_rate), init=weightinit))  # Fully connected layer
    model.add(Activation('relu'))  # Relu activation
    model.add(Dense(output_dim=outputdim, init=weightinit))
    model.add(Activation("softmax"))  # Final classification layer

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


def generate_CNN_hyperparameter_set(min_layers=1, max_layers=10,
                                    min_filters=10, max_filters=100,
                                    min_fc_nodes=10, max_fc_nodes=100):
    """ Generate a hyperparameter set that define a CNN model."""
    hyperparameters = generate_base_hyper_parameter_set()
    number_of_layers = np.random.randint(min_layers, max_layers)
    hyperparameters['filters'] = np.random.randint(
        min_filters, max_filters, number_of_layers)
    hyperparameters['fc_hidden_nodes'] = np.random.randint(
        min_fc_nodes, max_fc_nodes)
    return hyperparameters


def generate_DeepConvLSTM_hyperparameter_set(
    min_conv_layers=1, max_conv_layers=10,
    min_conv_filters=10, max_conv_filters=100,
    min_lstm_layers=1, max_lstm_layers=5,
        min_lstm_dims=10, max_lstm_dims=100):
    """ Generate a hyperparameter set that defines a DeepConvLSTM model."""
    hyperparameters = generate_base_hyper_parameter_set()
    number_of_conv_layers = np.random.randint(min_conv_layers, max_conv_layers)
    hyperparameters['filters'] = np.random.randint(
        min_conv_filters, max_conv_filters, number_of_conv_layers)
    number_of_lstm_layers = np.random.randint(min_lstm_layers, max_lstm_layers)
    hyperparameters['lstm_dims'] = np.random.randint(
        min_lstm_dims, max_lstm_dims, number_of_lstm_layers)
    return hyperparameters


def generate_base_hyper_parameter_set():
    """ Generate a base set of hyperparameters that are necessary for any model, but sufficient for none."""
    hyperparameters = {}
    hyperparameters['learning_rate'] = get_learning_rate()
    hyperparameters['regularization_rate'] = get_regularization()
    return hyperparameters


def get_learning_rate(low=1, high=4):
    """ Return random learning rate 10^-n where n is sampled uniformly between low and high bounds."""
    result = 10 ** (-np.random.uniform(low, high))
    return result


def get_regularization(low=1, high=4):
    """ Return random regularization rate 10^-n where n is sampled uniformly between low and high bounds."""
    return 10**(-np.random.uniform(low, high))
