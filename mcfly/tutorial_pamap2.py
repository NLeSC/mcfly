"""
 Summary:
 Function fetch_and_preprocess from tutorial_pamap2.py helps to fetch and
 preproces the data.
 Example function calls in 'Tutorial mcfly on PAMAP2.ipynb'
"""
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
    startpoints = np.where([1] + [labels[i] != labels[i-1] \
        for i in range(1, tot_len)])[0]
    endpoints = np.append(startpoints[1:]-1, tot_len-1)
    acts = [labels[s] for s,e in zip(startpoints, endpoints)]
    #Also split up the data, and only keep the non-zero activities
    Xysplit = [(X[s+borders:e-borders+1,:], a) \
        for s, e, a in zip(startpoints, endpoints, acts) if a != 0]
    Xysplit = [(X, y) for X, y in Xysplit if len(X)>0]
    Xlist = [X for X, y in Xysplit]
    ylist = [y for X, y in Xysplit]
    return Xlist, ylist

def sliding_window(frame_length, step, Xsamples, ysamples, ysampleslist, \
    Xsampleslist):
    """
    Splits time series in ysampleslist and Xsampleslist
    into segments by applying a sliding overlapping window
    of size equal to frame_length with steps equal to step
    it does this for all the samples and appends all the output together.
    So, the participant distinction is not kept
    """
    for j in range(len(Xsampleslist)):
        X = Xsampleslist[j]
        ybinary = ysampleslist[j]
        for i in range(0, X.shape[0]-frame_length, step):
            Xsub = X[i:i+frame_length,:]
            ysub = ybinary
            Xsamples.append(Xsub)
            ysamples.append(ysub)

def transform_y(y,mapclasses,nr_classes):
    """
    Transforms y, a tuple with sequences of class per time segment per sample,
    into a binary matrix per sample
    """
    ymapped = np.array([mapclasses[c] for c in y], dtype='int')
    ybinary = to_categorical(ymapped, nr_classes)
    return ybinary

def addheader(datasets):
    """
    The columns of the pandas data frame are numbers
    this function adds the column labels
    """
    axes = ['x', 'y', 'z']
    IMUsensor_columns = ['temperature'] + \
                    ['acc_16g_' + i for i in axes] + \
                    ['acc_6g_' + i for i in axes] + \
                    ['gyroscope_'+ i for i in axes] + \
                    ['magnometer_'+ i for i in axes] + \
                    ['orientation_' + str(i) for i in range(4)]
    header = ["timestamp", "activityID", "heartrate"] + ["hand_"+s \
        for s in IMUsensor_columns] \
        + ["chest_"+s for s in IMUsensor_columns]+ ["ankle_"+s \
            for s in IMUsensor_columns]
    for i in range(0,len(datasets)):
            datasets[i].columns = header
    return datasets

def split_dataset(datasets_filled,Xlists,ybinarylists):
    """
    This function split Xlists and ybinarylists into
    a train, test and val subset
    """
    train_range = slice(0, 6)
    val_range = 6
    test_range = slice(7,len(datasets_filled))
    Xtrainlist = [X for Xlist in Xlists[train_range] for X in Xlist]
    Xvallist = [X for X in Xlists[val_range]]
    Xtestlist = [X for Xlist in Xlists[test_range] for X in Xlist]
    ytrainlist = [y for ylist in ybinarylists[train_range] for y in ylist]
    yvallist = [y for y in ybinarylists[val_range]]
    ytestlist = [y for ylist in ybinarylists[test_range] for y in ylist]
    return Xtrainlist, Xvallist, Xtestlist, ytrainlist, yvallist, ytestlist

def numpify_and_store(x,y,Xname,yname,outdatapath,shuffle=False):
    """
    Converts python lists x and y into numpy arrays
    and stores the numpy array in directory outdatapath
    shuffle is optional and shuffles the samples
    """
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
    """
    Fetch the data and extract the contents of the zip file
    to the directory_to_extract_to.
    First check whether this was done before, if yes, then skip
    """
    targetdir = directory_to_extract_to + '/PAMAP2'
    if os.path.exists(targetdir):
        print('Data previously downloaded and stored in ' + targetdir)
    else:
        os.makedirs(targetdir) # create target directory
        #download the PAMAP2 data, this is 688 Mb
        path_to_zip_file = directory_to_extract_to + '/PAMAP2_Dataset.zip'
        test_file_exist = os.path.isfile(path_to_zip_file)
        if test_file_exist is False:
            url = str('https://archive.ics.uci.edu/ml/' +
                'machine-learning-databases/00231/PAMAP2_Dataset.zip')
            #retrieve data from url
            local_fn, headers = urllib.request.urlretrieve(url,\
                filename=path_to_zip_file)
            print('Download complete and stored in: ' + path_to_zip_file)
        else:
            print('The data was previously downloaded and stored in ' +
                path_to_zip_file)
        # unzip
        with zipfile.ZipFile(path_to_zip_file ,"r") as zip_ref:
            zip_ref.extractall(targetdir)
    return targetdir


def fetch_and_preprocess(directory_to_extract_to, columns_to_use=None):
    """
    High level function to fetch_and_preprocess the PAMAP2 dataset
    directory_to_extract_to: the directory where the data will be stored
    columns_to_use: the columns to use
    """
    if columns_to_use is None:
        columns_to_use = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
                     'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
                     'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z']
    targetdir = fetch_data(directory_to_extract_to)
    outdatapath = targetdir + '/PAMAP2_Dataset' + '/slidingwindow512cleaned/'
    if not os.path.exists(outdatapath):
        os.makedirs(outdatapath)
    if os.path.isfile(outdatapath+'X_train.npy'):
        print('Data previously pre-processed and np-files saved to ' +
            outdatapath)
    else:
        datadir = targetdir + '/PAMAP2_Dataset/Protocol'
        filenames = listdir(datadir)
        print('Start pre-processing all ' + str(len(filenames)) + ' files...')
        # load the files and put them in a list of pandas dataframes:
        datasets = [pd.read_csv(datadir+'/'+fn, header=None, sep=' ') \
            for fn in filenames]
        datasets = addheader(datasets) # add headers to the datasets
        #Interpolate dataset to get same sample rate between channels
        datasets_filled = [d.interpolate() for d in datasets]
        # Create mapping for class labels
        ysetall = [set(np.array(data.activityID)) - set([0]) \
            for data in datasets_filled]
        classlabels = list(set.union(*[set(y) for y in ysetall]))
        nr_classes = len(classlabels)
        mapclasses = {classlabels[i] : i for i in range(len(classlabels))}
        #Create input (X) and output (y) sets
        Xall = [np.array(data[columns_to_use]) for data in datasets_filled]
        yall = [np.array(data.activityID) for data in datasets_filled]
        Xylists = [split_activities(y, X) for X, y in zip(Xall, yall)]
        Xlists, ylists = zip(*Xylists)
        ybinarylists = [transform_y(y, mapclasses, nr_classes) for y in ylists]
        # Split in train, test and val
        Xtrainlist, Xvallist, Xtestlist, ytrainlist, yvallist, ytestlist = \
            split_dataset(datasets_filled, Xlists, ybinarylists)
        # Take sliding-window frames. Target is label of last time step
        # Data is 100 Hz
        frame_length = int(5.12 * 100)
        step = 1 * 100
        Xtrain = []
        ytrain = []
        Xval = []
        yval = []
        Xtest = []
        ytest = []
        sliding_window(frame_length, step, Xtrain, ytrain, ytrainlist, \
            Xtrainlist)
        sliding_window(frame_length, step, Xval, yval, yvallist, Xvallist)
        sliding_window(frame_length, step, Xtest, ytest, ytestlist, Xtestlist)
        numpify_and_store(Xtrain, ytrain, 'X_train', 'y_train', outdatapath, \
            shuffle=True)
        numpify_and_store(Xval, yval, 'X_val', 'y_val', outdatapath, \
            shuffle=True)
        numpify_and_store(Xtest, ytest, 'X_test', 'y_test', outdatapath, \
            shuffle=True)
        print('Processed data succesfully stored in ' + outdatapath)
    return outdatapath

def load_data(outputpath):
    ext = '.npy'
    Xtrain = np.load(outputpath+'X_train'+ext)
    ytrain_binary = np.load(outputpath+'y_train'+ext)
    Xval = np.load(outputpath+'X_val'+ext)
    yval_binary = np.load(outputpath+'y_val'+ext)
    Xtest = np.load(outputpath+'X_test'+ext)
    ytest_binary = np.load(outputpath+'y_test'+ext)
    return Xtrain, ytrain_binary, Xval, yval_binary, Xtest, ytest_binary
