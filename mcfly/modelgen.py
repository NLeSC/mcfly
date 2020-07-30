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

import numpy as np
from mcfly.models import Model_CNN, Model_ConvLSTM, Model_ResNet, Model_InceptionTime


def generate_models(x_shape,
                    number_of_classes,
                    number_of_models,
                    model_types=['CNN', 'DeepConvLSTM', 'ResNet', 'InceptionTime'],
                    metrics=['accuracy'],
                    **hyperparameter_ranges):
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
    model_types : list, optional
        Type of model to build: 'CNN' 'DeepConvLSTM', 'ResNet', or 'InceptionTime'.
        Default is to use all models equally frequently.
    metrics : list
        Metrics to calculate on the validation set.
        See https://keras.io/metrics/ for possible values.
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

    Further hyperparameter ranges can be specified according to the respective
    model types used. #TODO: add link to model class docstrings

    Returns
    -------
    models : list
        List of compiled models
    """
    # Set default hyperparameter ranges
    defaults = {'low_lr': 1,
                'high_lr': 4,
                'low_reg': 1,
                'high_reg': 4}

    default_models = {'CNN': Model_CNN,
                      'DeepConvLSTM': Model_ConvLSTM,
                      'ResNet': Model_ResNet,
                      'InceptionTime': Model_InceptionTime}

    # Replace default hyperparameter ranges with input
    for key, value in hyperparameter_ranges.items():
        if key in defaults:
            print("The value of {} is set from {} (default) to {}".format(key, defaults[key], value))
            defaults[key] = value

    # Add missing parameters from default
    for key, value in defaults.items():
        if key not in hyperparameter_ranges:
            hyperparameter_ranges[key] = value

    model_types_selected = []

    if number_of_models%len(model_types) > 0:
        number_of_models = \
            int(np.ceil(number_of_models/len(model_types)))*len(model_types)
        print("To have equal number of models for all types, number of models was changed to {}".format(number_of_models))

    for i in range(number_of_models // len(model_types)):
        np.random.shuffle(model_types)
        model_types_selected.extend(model_types)

    # Create list of Keras models and their hyperparameters
    # -------------------------------------------------------------------------
    models = []
    for current_model_type in model_types_selected:
        if current_model_type in default_models:
            model_type = default_models[current_model_type](x_shape, number_of_classes,
                                                            metrics, **hyperparameter_ranges)
            hyperparameters = model_type.generate_hyperparameters()
            model = model_type.create_model(**hyperparameters)
            model_name = current_model_type
        elif isinstance(current_model_type, str):
            print("Unknown model name given.")
            break
        else: # Assume model class was passed
            model_type = current_model_type(x_shape, number_of_classes,
                                            metrics, **hyperparameter_ranges)
            hyperparameters = model_type.generate_hyperparameters()
            model = model_type.create_model(**hyperparameters)
            model_name = model_type.model_name

        models.append((model, hyperparameters, model_name))
    return models
