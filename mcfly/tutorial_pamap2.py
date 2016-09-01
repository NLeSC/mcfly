#import required python modules
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
import os.path
import urllib.request
import zipfile
import keras
from keras.utils.np_utils import to_categorical

def split_activities(labels, X, borders=10*100):
    """
    Splits up the data per activity and exclude activity=0.
    Also remove borders for each activity.
    Returns lists with subdatasets
    """
    tot_len = len(labels)
    startpoints = np.where([1] + [labels[i]!=labels[i-1] for i in range(1, tot_len)])[0]
    endpoints = np.append(startpoints[1:]-1, tot_len-1)
    acts = [labels[s] for s,e in zip(startpoints, endpoints)]
    #Also split up the data, and only keep the non-zero activities
    Xy_split = [(X[s+borders:e-borders+1,:], a) for s,e,a in zip(startpoints, endpoints, acts) if a != 0]
    Xy_split = [(X, y) for X,y in Xy_split if len(X)>0]
    X_list = [X for X,y in Xy_split]
    y_list = [y for X,y in Xy_split]
    return X_list, y_list

def sliding_window(frame_length, step, X_samples, y_samples,y_samples_list,X_samples_list):
    for j in range(len(X_samples_list)):
        X = X_samples_list[j]
        y_binary = y_samples_list[j]
        for i in range(0, X.shape[0]-frame_length, step):
            X_sub = X[i:i+frame_length,:]
            y_sub = y_binary
            X_samples.append(X_sub)
            y_samples.append(y_sub)

def transform_y(y,mapclasses,nr_classes):
    y_mapped = np.array([mapclasses[c] for c in y], dtype='int')
    y_binary = to_categorical(y_mapped, nr_classes)
    return y_binary

def addheader(datasets):
    # The columns are numbers, which is not very practical. Let's add column labels to the pandas dataframe:
    axes = ['x', 'y', 'z']
    IMUsensor_columns = ['temperature'] + \
                    ['acc_16g_' + i for i in axes] + \
                    ['acc_6g_' + i for i in axes] + \
                    ['gyroscope_'+ i for i in axes] + \
                    ['magnometer_'+ i for i in axes] + \
                    ['orientation_' + str(i) for i in range(4)]
    header = ["timestamp", "activityID", "heartrate"] + ["hand_"+s for s in IMUsensor_columns]\
        + ["chest_"+s for s in IMUsensor_columns]+ ["ankle_"+s for s in IMUsensor_columns]
    for i in range(0,len(datasets)):
            datasets[i].columns = header
    return datasets

def split_dataset(datasets_filled,X_lists,y_binary_lists):
    # Split in train, test and val
    train_range = slice(0, 6)
    val_range = 6
    test_range = slice(7,len(datasets_filled))
    X_train_list = [X for X_list in X_lists[train_range] for X in X_list]
    X_val_list = [X for X in X_lists[val_range]]
    X_test_list = [X for X_list in X_lists[test_range] for X in X_list]
    y_train_list = [y for y_list in y_binary_lists[train_range] for y in y_list]
    y_val_list = [y for y in y_binary_lists[val_range]]
    y_test_list = [y for y_list in y_binary_lists[test_range] for y in y_list]
    return X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list

def numpify_and_store(x,y,Xname,yname,outdatapath,shuffle=False):
    x = np.array(x)
    y = np.array(y)
    #Shuffle around the train set
    if shuffle is True:
        np.random.seed(123)
        neworder = np.random.permutation(x.shape[0])
        x = x[neworder,:,:]
        y = y[neworder,:]
    # Save binary file
    np.save(outdatapath+ Xname, x)
    np.save(outdatapath+ yname, y)


def fetch_data(directory_to_extract_to):
    targetdir = directory_to_extract_to + '/PAMAP2'
    if os.path.exists(targetdir):
        print('Data previously downloaded and stored in ' + targetdir)
    else:
        os.makedirs(targetdir) # create target directory
        #download the PAMAP2 data, this is 688 Mb
        path_to_zip_file = directory_to_extract_to + '/PAMAP2_Dataset.zip'
        test_file_exist = os.path.isfile(path_to_zip_file)
        if test_file_exist is False:
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip'
            local_fn, headers = urllib.request.urlretrieve(url,filename=path_to_zip_file) #retrieve data from url
            print('Download complete and stored in: ' + path_to_zip_file )
        else:
            print('The data was previously downloaded and stored in ' + path_to_zip_file )
        # unzip
        with zipfile.ZipFile(path_to_zip_file ,"r") as zip_ref:
            zip_ref.extractall(targetdir)
    return targetdir


def fetch_and_preprocess(directory_to_extract_to,columns_to_use):
    targetdir = fetch_data(directory_to_extract_to)
    outdatapath = targetdir + '/PAMAP2_Dataset' + '/slidingwindow512cleaned/'
    if not os.path.exists(outdatapath):
        os.makedirs(outdatapath)
    if os.path.isfile(outdatapath+'X_train.npy'):
        print('Data previously pre-processed and np-files saved to ' + outdatapath)
    else:
        datadir = targetdir + '/PAMAP2_Dataset/Protocol'
        filenames = listdir(datadir)
        print('Start pre-processing all ' + str(len(filenames)) + ' files...')
        # load the files and put them in a list of pandas dataframes:
        datasets = [pd.read_csv(datadir+'/'+fn, header=None, sep=' ') for fn in filenames]
        datasets = addheader(datasets) # add headers to the datasets
        datasets_filled = [d.interpolate() for d in datasets] #Interpolate dataset to get same sample rate between channels
        # Create mapping for class labels
        y_set_all = [set(np.array(data.activityID)) - set([0]) for data in datasets_filled]
        classlabels = list(set.union(*[set(y) for y in y_set_all]))
        nr_classes = len(classlabels)
        mapclasses = {classlabels[i] : i for i in range(len(classlabels))}
        #Create input (X) and output (y) sets
        X_all = [np.array(data[columns_to_use]) for data in datasets_filled]
        y_all = [np.array(data.activityID) for data in datasets_filled]
        Xy_lists = [split_activities(y, X) for X,y in zip(X_all, y_all)]
        X_lists, y_lists = zip(*Xy_lists)
        y_binary_lists = [transform_y(y,mapclasses,nr_classes) for y in y_lists]
        # Split in train, test and val
        X_train_list, X_val_list, X_test_list, y_train_list, y_val_list, y_test_list = split_dataset(datasets_filled,X_lists,y_binary_lists)
        # Take sliding-window frames. Target is label of last time step
        # Data is 100 Hz
        frame_length = int(5.12 * 100)
        step = 1 * 100
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        sliding_window(frame_length, step, X_train, y_train,y_train_list,X_train_list)
        sliding_window(frame_length, step, X_val, y_val,y_val_list,X_val_list)
        sliding_window(frame_length, step, X_test, y_test,y_test_list,X_test_list)
        numpify_and_store(X_train,y_train,'X_train','y_train',outdatapath,shuffle=True)
        numpify_and_store(X_val,y_val,'X_val','y_val',outdatapath,shuffle=True)
        numpify_and_store(X_test,y_test,'X_test','y_test',outdatapath,shuffle=True)
        print('Processed data succesfully stored in ' + outdatapath)
    return outdatapath

def load_data(outputpath):
    ext = '.npy'
    X_train = np.load(outputpath+'X_train'+ext)
    y_train_binary = np.load(outputpath+'y_train_binary'+ext)
    X_val = np.load(outputpath+'X_val'+ext)
    y_val_binary = np.load(outputpath+'y_val_binary'+ext)
    X_test = np.load(outputpath+'X_test'+ext)
    y_test_binary = np.load(outputpath+'y_test_binary'+ext)
    return X_train, y_train_binary, X_val, y_val_binary, X_test, y_test_binary
