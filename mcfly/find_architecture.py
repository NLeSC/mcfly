"""
 Summary:
 This module provides the main functionality of mcfly: searching for an
 optimal model architecture. The work flow is as follows:
 Function generate_models from modelgen.py generates and compiles models.
 Function train_models_on_samples trains those models.
 Function plotTrainingProcess plots the training process.
 Function find_best_architecture is wrapper function that combines
 these steps.
 Example function calls can be found in the tutorial notebook
 'EvaluateDifferentModels.ipynb'.
"""
import numpy as np
from matplotlib import pyplot as plt
from . import modelgen
from sklearn import neighbors, metrics
import warnings
import json
import os
from keras.callbacks import EarlyStopping

def train_models_on_samples(X_train, y_train, X_val, y_val, models,
                            nr_epochs=5, subset_size=100, verbose=True,
                            outputfile=None, early_stopping=False):
    """
    Given a list of compiled models, this function trains
    them all on a subset of the train data. If the given size of the subset is
    smaller then the size of the data, the complete data set is used.

    Parameters
    ----------
    X_train : numpy array of shape (num_samples, num_timesteps, num_channels)
        The input dataset for training
    y_train : numpy array of shape (num_samples, num_classes)
        The output classes for the train data, in binary format
    X_val : numpy array of shape (num_samples_val, num_timesteps, num_channels)
        The input dataset for validation
    y_val : numpy array of shape (num_samples_val, num_classes)
        The output classes for the validation data, in binary format
    models : list of model, params, modeltypes
        List of keras models to train
    nr_epochs : int, optional
        nr of epochs to use for training one model
    subset_size :
        The number of samples used from the complete train set
    verbose : bool, optional
        flag for displaying verbose output
    outputfile : str, optional
        File location to store the model results
    early_stopping: bool
        Stop when validation loss does not decrease

    Returns
    ----------
    histories : list of Keras History objects
        train histories for all models
    val_accuracies : list of floats
        validation accuraracies of the models
    val_losses : list of floats
        validation losses of the models
    """
    # if subset_size is smaller then X_train, this will work fine
    X_train_sub = X_train[:subset_size, :, :]
    y_train_sub = y_train[:subset_size, :]

    histories = []
    val_accuracies = []
    val_losses = []
    for i, (model, params, model_types) in enumerate(models):
        if verbose:
            print('Training model %d' % i, model_types)
        if early_stopping:
            callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=verbose, mode='auto')]
        else:
            callbacks = []
        history = model.fit(X_train_sub, y_train_sub,
                            nb_epoch=nr_epochs, batch_size=20,
                            # see comment on subsize_set
                            validation_data=(X_val, y_val),
                            verbose=verbose,
                            callbacks=callbacks)
        histories.append(history)
        val_accuracies.append(history.history['val_acc'][-1])
        val_losses.append(history.history['val_loss'][-1])
        if outputfile is not None:
            store_train_hist_as_json(params, model_types,
                                     history.history, outputfile)
    return histories, val_accuracies, val_losses


def store_train_hist_as_json(params, model_type, history, outputfile):
    """
    This function stores the model parameters, the loss and accuracy history
    of one model in a JSON file. It appends the model information to the
    existing models in the file.

    Parameters
    ----------
    params : dict
        parameters for one model
    model_type : Keras model object
        Keras model object for one model
    history : dict
        training history from one model
    outputfile : str
        path where the json file needs to be stored
    """
    jsondata = params.copy()
    for k in jsondata.keys():
        if isinstance(jsondata[k], np.ndarray):
            jsondata[k] = jsondata[k].tolist()
    jsondata['train_acc'] = history['acc']
    jsondata['train_loss'] = history['loss']
    jsondata['val_acc'] = history['val_acc']
    jsondata['val_loss'] = history['val_loss']
    jsondata['modeltype'] = model_type
    jsondata['modeltype'] = model_type
    if os.path.isfile(outputfile):
        with open(outputfile, 'r') as outfile:
            previousdata = json.load(outfile)
    else:
        previousdata = []
    previousdata.append(jsondata)
    with open(outputfile, 'w') as outfile:
        json.dump(previousdata, outfile, sort_keys=True,
                  indent=4, ensure_ascii=False)


def plotTrainingProcess(history, name='Model', ax=None):
    """
    This function plots the loss and accuracy on the train and validation set,
    for each epoch in the history of one model.

    Parameters
    ----------
    history : keras History object
        The history object of the training process corresponding to one model
    name : str
        Name of the model, to display in the title
    ax : Axis, optional
        Specific axis to plot on

    """
    if ax is None:
        fig, ax = plt.subplots()
    ax2 = ax.twinx()
    LN = len(history.history['val_loss'])
    val_loss, = ax.plot(range(LN), history.history['val_loss'], 'g--',
                        label='validation loss')
    train_loss, = ax.plot(range(LN), history.history['loss'], 'g-',
                          label='train loss')
    val_acc, = ax2.plot(range(LN), history.history['val_acc'], 'b--',
                        label='validation accuracy')
    train_acc, = ax2.plot(range(LN), history.history['acc'], 'b-',
                          label='train accuracy')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss', color='g')
    ax2.set_ylabel('accuracy', color='b')
    plt.legend(handles=[val_loss, train_loss, val_acc, train_acc],
               loc=2, bbox_to_anchor=(1.1, 1))
    plt.title(name)


def find_best_architecture(X_train, y_train, X_val, y_val, verbose=True,
                           number_of_models=5, nr_epochs=5, subset_size=100,
                           outputpath=None, **kwargs
                           ):
    """
    Tries out a number of models on a subsample of the data,
    and outputs the best found architecture and hyperparameters.

    Parameters
    ----------
    X_train : numpy array
        The input dataset for training of shape
        (num_samples, num_timesteps, num_channels)
    y_train : numpy array
        The output classes for the train data, in binary format of shape
        (num_samples, num_classes)
    X_val : numpy array
        The input dataset for validation of shape
        (num_samples_val, num_timesteps, num_channels)
    y_val : numpy array
        The output classes for the validation data, in binary format of shape
        (num_samples_val, num_classes)
    verbose : bool, optional
        flag for displaying verbose output
    number_of_models : int, optiona
        The number of models to generate and test
    nr_epochs : int, optional
        The number of epochs that each model is trained
    subset_size : int, optional
        The size of the subset of the data that is used for finding
        the optimal architecture
    outputpath : str, optional
        File location to store the model results
    **kwargs: key-value parameters
        parameters for generating the models
        (see docstring for modelgen.generate_models)

    Returns
    ----------
    best_model : Keras model
        Best performing model, already trained on a small sample data set.
    best_params : dict
        Dictionary containing the hyperparameters for the best model
    best_model_type : str
        Type of the best model
    knn_acc : float
        accuaracy for kNN prediction on validation set
    """
    models = modelgen.generate_models(X_train.shape, y_train.shape[1],
                                      number_of_models=number_of_models,
                                      **kwargs)
    histories, val_accuracies, val_losses = train_models_on_samples(X_train,
                                                                    y_train,
                                                                    X_val,
                                                                    y_val,
                                                                    models,
                                                                    nr_epochs,
                                                                    subset_size=subset_size,
                                                                    verbose=verbose,
                                                                    outputfile=outputpath)
    best_model_index = np.argmax(val_accuracies)
    best_model, best_params, best_model_type = models[best_model_index]
    knn_acc = kNN_accuracy(
        X_train[:subset_size, :, :], y_train[:subset_size, :], X_val, y_val)
    if verbose:
        for i in range(len(models)):  # now one plot per model, ultimately we
            # may want all models in one plot to allow for direct comparison
            name = str(models[i][1])
            plotTrainingProcess(histories[i], name)
        print('Best model: model ', best_model_index)
        print('Model type: ', best_model_type)
        print('Hyperparameters: ', best_params)
        print('Accuracy on validation set: ', val_accuracies[best_model_index])
        print('Accuracy of kNN on validation set', knn_acc)

    if val_accuracies[best_model_index] < knn_acc:
        warnings.warn('Best model not better than kNN: ' +
                      str(val_accuracies[best_model_index]) + ' vs  ' +
                      str(knn_acc)
                      )
    return best_model, best_params, best_model_type, knn_acc


def kNN_accuracy(X_train, y_train, X_val, y_val, k=1):
    """
    Performs k-Neigherst Neighbors and returns the accuracy score.

    Parameters
    ----------
    X_train : numpy array
        Train set of shape (num_samples, num_timesteps, num_channels)
    y_train : numpy array
        Class labels for train set
    X_val : numpy array
        Validation set of shape (num_samples, num_timesteps, num_channels)
    y_val : numpy array
        Class labels for validation set
    k : int
        number of neighbors to use for classifying

    Returns
    -------
    accuracy: float
        accuracy score on the validation set
    """
    num_samples, num_timesteps, num_channels = X_train.shape
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(
        X_train.reshape(
            num_samples,
            num_timesteps *
            num_channels),
        y_train)
    num_samples, num_timesteps, num_channels = X_val.shape
    val_predict = clf.predict(
        X_val.reshape(num_samples,
                      num_timesteps * num_channels))
    return metrics.accuracy_score(val_predict, y_val)
