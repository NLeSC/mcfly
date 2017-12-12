"""
 Summary:
 Functions to save and store a model. The current keras
 function to do this does not work in python3. Therefore, we
 implemented our own functions until the keras functionality has matured.
 Example function calls in 'Tutorial mcfly on PAMAP2.ipynb'
"""
from keras.models import model_from_json
import keras

import json
import numpy as np
import os
import uuid
from collections import namedtuple


TrainedModel = namedtuple(
        'TrainedModel', ['history', 'model'])


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
    json_string = model.to_json()  # save architecture to json string
    json_path = os.path.join(filepath, modelname + '_architecture.json')
    with open(json_path, 'w') as outfile:
        json.dump(json_string, outfile, sort_keys=True, indent=4,
                  ensure_ascii=False)
    wweights = model.get_weights()  # get weight from model
    numpy_path = os.path.join(filepath, modelname + '_weights')
    np.save(numpy_path,
            wweights)  # save weights in npy file
    return json_path, numpy_path


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
    model_repro : Keras object
        reproduced model
    """
    with open(os.path.join(filepath, modelname + '_architecture.json'), 'r') as outfile:
        json_string_loaded = json.load(outfile)
    model_repro = model_from_json(json_string_loaded)
    # wweights2 = model_repro.get_weights()
    #  extracting the weights would give us the untrained/default weights
    wweights_recovered = np.load(
        os.path.join(filepath, modelname + '_weights.npy'))  # load the original weights
    model_repro.set_weights(wweights_recovered)  # now set the weights
    return model_repro

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


try:
    import noodles
    from noodles.serial.numpy import arrays_to_string
    from noodles.serial.namedtuple import SerNamedTuple


    class SerModel(noodles.serial.Serialiser):
        def __init__(self):
            super(SerModel, self).__init__(keras.models.Model)

        def encode(self, obj, make_rec):
            random_filename = str(uuid.uuid4()) + '.hdf5'
            obj.save(random_filename)
            return make_rec({'filename': random_filename},
                            files=[random_filename], ref=True)

        def decode(self, cls, data):
            return keras.models.load_model(data['filename'])


    def serial_registry():
        return noodles.serial.Registry(
            # parent=noodles.serial.pickle() +
            parent=noodles.serial.base() + arrays_to_string(),
            types={
                keras.models.Model: SerModel(),
                TrainedModel: SerNamedTuple(TrainedModel)
            }
        )

except ImportError:
    pass
