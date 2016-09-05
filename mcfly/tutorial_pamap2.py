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
    xysplit = [(X[s+borders:e-borders+1, :], a) \
        for s, e, a in zip(startpoints, endpoints, acts) if a != 0]
    xysplit = [(X, y) for X, y in xysplit if len(X) > 0]
    Xlist = [X for X, y in xysplit]
    ylist = [y for X, y in xysplit]
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
        for i in range(0, x.shape[0]-frame_length, step):
            xsub = X[i:i+frame_length, :]
            ysub = ybinary
            Xsamples.append(xsub)
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
    This function split xlists and ybinarylists into
    a train, test and val subset
    """
    train_range = slice(0, 6)
    val_range = 6
    test_range = slice(7,len(datasets_filled))
    x_trainlist = [x for xlist in Xlists[train_range] for x in Xlist]
    x_vallist = [x for x in Xlists[val_range]]
    x_testlist = [x for xlist in Xlists[test_range] for x in Xlist]
    y_trainlist = [y for ylist in ybinarylists[train_range] for y in ylist]
    y_vallist = [y for y in ybinarylists[val_range]]
    y_testlist = [y for ylist in ybinarylists[test_range] for y in ylist]
    return x_trainlist, x_vallist, x_testlist, y_trainlist, \
        y_vallist, y_testlist

def numpify_and_store(X, y, xname, yname, outdatapath, shuffle=False):
    """
    Converts python lists x and y into numpy arrays
    and stores the numpy array in directory outdatapath
    shuffle is optional and shuffles the samples
    """
    X = np.array(X)
    y = np.array(y)
    #Shuffle around the train set
    if shuffle is True:
        np.random.seed(123)
        neworder = np.random.permutation(X.shape[0])
        X = X[neworder, :, :]
        y = y[neworder, :]
    # Save binary file
    np.save(outdatapath+ xname, X)
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
    if os.path.isfile(outdatapath+'x_train.npy'):
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
        #Create input (x) and output (y) sets
        xall = [np.array(data[columns_to_use]) for data in datasets_filled]
        yall = [np.array(data.activityID) for data in datasets_filled]
        xylists = [split_activities(y, x) for x, y in zip(xall, yall)]
        xlists, ylists = zip(*xylists)
        ybinarylists = [transform_y(y, mapclasses, nr_classes) for y in ylists]
        # Split in train, test and val
        x_trainlist, x_vallist, x_testlist, y_trainlist, y_vallist, y_testlist =\
            split_dataset(datasets_filled, xlists, ybinarylists)
        # Take sliding-window frames. Target is label of last time step
        # Data is 100 Hz
        frame_length = int(5.12 * 100)
        step = 1 * 100
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        sliding_window(frame_length, step, x_train, y_train, y_trainlist, \
            x_trainlist)
        sliding_window(frame_length, step, x_val, y_val, y_vallist, x_vallist)
        sliding_window(frame_length, step, x_test, y_test, y_testlist, x_testlist)
        numpify_and_store(x_train, y_train, 'X_train', 'y_train', outdatapath, \
            shuffle=True)
        numpify_and_store(x_val, y_val, 'X_val', 'y_val', outdatapath, \
            shuffle=False)
        numpify_and_store(x_test, y_test, 'X_test', 'y_test', outdatapath, \
            shuffle=False)
        print('Processed data succesfully stored in ' + outdatapath)
    return outdatapath

def load_data(outputpath):
    ext = '.npy'
    x_train = np.load(outputpath+'X_train'+ext)
    y_train_binary = np.load(outputpath+'y_train'+ext)
    x_val = np.load(outputpath+'X_val'+ext)
    y_val_binary = np.load(outputpath+'y_val'+ext)
    x_test = np.load(outputpath+'X_test'+ext)
    y_test_binary = np.load(outputpath+'y_test'+ext)
    return x_train, y_train_binary, x_val, y_val_binary, x_test, y_test_binary
