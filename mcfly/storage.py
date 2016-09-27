"""
 Summary:
 Functions to save and store a model. The current keras
 function to do this does not work in python3. Therefore, we
 implemented our own functions until the keras functionality has matured.
 Example function calls in 'Tutorial mcfly on PAMAP2.ipynb'
"""
from keras.models import model_from_json
import json
import pickle
import numpy as np

def savemodel(model,filepath,modelname):
    """ Save model + weights + params TO json + npy + pkl file, respectively
    Input:
    - model (Keras object)
    - filepath: directory where the data will be stored
    - modelname: name of the model to be used in the filename
    """
    json_string = model.to_json() # save architecture to json string
    with open(filepath + modelname + '_architecture.json', 'w') as outfile:
        json.dump(json_string, outfile, sort_keys = True, indent = 4, ensure_ascii=False)
    wweights = model.get_weights() #get weight from model
    np.save(filepath+modelname+'_weights',wweights) #save weights in npy file
    return None

def loadmodel(filepath,modelname):
    """ Load model + weights FROM json + npy file, respectively
    Input:
    - filepath: directory where the data will be stored
    - modelname: name of the model to be used in the filename
    """
    with open(filepath + modelname + '_architecture.json', 'r') as outfile:
         json_string_loaded = json.load(outfile)
    model_repro = model_from_json(json_string_loaded)
    wweights2 = model_repro.get_weights() # extracting the weights would give us the untrained/default weights
    wweights_recovered =np.load(filepath+modelname+'_weights.npy') #load the original weights
    model_repro.set_weights(wweights_recovered) # now set the weights
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
