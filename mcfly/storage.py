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

"""
 Summary:
 Functions to save and store a model. The current keras
 function to do this does not work in python3. Therefore, we
 implemented our own functions until the keras functionality has matured.
 Example function calls in 'Tutorial mcfly on PAMAP2.ipynb'
"""
from tensorflow.keras.models import model_from_json
import json
import numpy as np
import os


def savemodel(model, filepath, modelname):
    """ Save model  to json file and weights to npy file

    Parameters
    ----------
    model : Keras object
        model to save
    filepath : str
        directory where the data will be stored
    modelname : str
        name of the model to be used in the filename

    Returns
    ----------
    json_path : str
        Path to json file with architecture
    numpy_path : str
        Path to npy file with weights
    """
    json_path = _save_architecture_and_return_path(filepath, model, modelname)
    numpy_path = _save_weights_and_return_path(filepath, model, modelname)
    return json_path, numpy_path


def _save_weights_and_return_path(filepath, model, modelname):
    numpy_path = os.path.join(filepath, modelname + '_weights')
    np.save(numpy_path, model.get_weights())
    return numpy_path


def _save_architecture_and_return_path(filepath, model, modelname):
    json_path = os.path.join(filepath, modelname + '_architecture.json')
    with open(json_path, 'w') as outfile:
        json.dump(model.to_json(), outfile, sort_keys=True, indent=4, ensure_ascii=False)
    return json_path


def loadmodel(filepath, modelname):
    """ Load model + weights from json + npy file, respectively

    Parameters
    ----------
    filepath : str
        directory where the data will be stored
    modelname : str
        name of the model to be used in the filename

    Returns
    ----------
    model : Keras object
        reproduced model
    """
    with open(os.path.join(filepath, modelname + '_architecture.json'), 'r') as outfile:
        loaded_json = json.load(outfile)
    model = model_from_json(loaded_json)
    weights_path = os.path.join(filepath, modelname + '_weights.npy')
    model.set_weights(np.load(weights_path, allow_pickle=True))
    return model

# If we would use standard Keras function, which stores model and weights
# in HDF5 format it would look like code below. However, we did not use this
# because
# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
# it is not compatible with default Keras version in python3.
# from keras.models import load_model
# import h5py
# modelh5=models[0]
# modelh5.save(resultpath+'mymodel.h5')
# del modelh5
# modelh5 = load_model(resultpath+'mymodel.h5')
