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
import zipfile
import keras
from keras.utils.np_utils import to_categorical
import sys
import six.moves.urllib as urllib


def split_activities(labels, X, borders=10 * 100):
    """
    Splits up the data per activity and exclude activity=0.
    Also remove borders for each activity.
    Returns lists with subdatasets

    Parameters
    ----------
    labels : numpy array
        Activity labels
    X : numpy array
        Data points
    borders : int
        Nr of timesteps to remove from the borders of an activity

    Returns
    -------
    X_list, y_list
    """
    tot_len = len(labels)
    startpoints = np.where([1] + [labels[i] != labels[i - 1]
                                  for i in range(1, tot_len)])[0]
    endpoints = np.append(startpoints[1:] - 1, tot_len - 1)
    acts = [labels[s] for s, e in zip(startpoints, endpoints)]
    # Also split up the data, and only keep the non-zero activities
    xysplit = [(X[s + borders:e - borders + 1, :], a)
               for s, e, a in zip(startpoints, endpoints, acts) if a != 0]
    xysplit = [(X, y) for X, y in xysplit if len(X) > 0]
    Xlist = [X for X, y in xysplit]
    ylist = [y for X, y in xysplit]
    return Xlist, ylist


def sliding_window(frame_length, step, Xsamples,
                   ysamples, Xsampleslist, ysampleslist):
    """
    Splits time series in ysampleslist and Xsampleslist
    into segments by applying a sliding overlapping window
    of size equal to frame_length with steps equal to step
    it does this for all the samples and appends all the output together.
    So, the participant distinction is not kept

    Parameters
    ----------
    frame_length : int
        Length of sliding window
    step : int
        Stepsize between windows
    Xsamples : list
        Existing list of window fragments
    ysamples : list
        Existing list of window fragments
    Xsampleslist : list
        Samples to take sliding windows from
    ysampleslist
        Samples to take sliding windows from

    """
    for j in range(len(Xsampleslist)):
        X = Xsampleslist[j]
        ybinary = ysampleslist[j]
        for i in range(0, X.shape[0] - frame_length, step):
            xsub = X[i:i + frame_length, :]
            ysub = ybinary
            Xsamples.append(xsub)
            ysamples.append(ysub)


def transform_y(y, mapclasses, nr_classes):
    """
    Transforms y, a list with one sequence of A timesteps
    and B unique classes into a binary Numpy matrix of
    shape (A, B)

    Parameters
    ----------
    y : list or array
        List of classes
    mapclasses : dict
        dictionary that maps the classes to numbers
    nr_classes : int
        total number of classes
    """
    ymapped = np.array([mapclasses[c] for c in y], dtype='int')
    ybinary = to_categorical(ymapped, nr_classes)
    return ybinary


def addheader(datasets):
    """
    The columns of the pandas data frame are numbers
    this function adds the column labels

    Parameters
    ----------
    datasets : list
        List of pandas dataframes
    """
    axes = ['x', 'y', 'z']
    IMUsensor_columns = ['temperature'] + \
        ['acc_16g_' + i for i in axes] + \
        ['acc_6g_' + i for i in axes] + \
        ['gyroscope_' + i for i in axes] + \
        ['magnometer_' + i for i in axes] + \
        ['orientation_' + str(i) for i in range(4)]
    header = ["timestamp", "activityID", "heartrate"] + ["hand_" + s
                                                         for s in IMUsensor_columns] \
        + ["chest_" + s for s in IMUsensor_columns] + ["ankle_" + s
                                                       for s in IMUsensor_columns]
    for i in range(0, len(datasets)):
        datasets[i].columns = header
    return datasets


def numpify_and_store(X, y, xname, yname, outdatapath, shuffle=False):
    """
    Converts python lists x 3D and y 1D into numpy arrays
    and stores the numpy array in directory outdatapath
    shuffle is optional and shuffles the samples

    Parameters
    ----------
    X : list
        list with data
    y : list
        list with data
    xname : str
        name to store the x arrays
    yname : str
        name to store the y arrays
    outdatapath : str
        path to the directory to store the data
    shuffle : bool
        whether to shuffle the data before storing
    """
    X = np.array(X)
    y = np.array(y)
    # Shuffle the train set
    if shuffle is True:
        np.random.seed(123)
        neworder = np.random.permutation(X.shape[0])
        X = X[neworder, :, :]
        y = y[neworder, :]
    # Save binary file
    np.save(outdatapath + xname, X)
    np.save(outdatapath + yname, y)


def fetch_data(directory_to_extract_to):
    """
    Fetch the data and extract the contents of the zip file
    to the directory_to_extract_to.
    First check whether this was done before, if yes, then skip

    Parameters
    ----------
    directory_to_extract_to : str
        directory to create subfolder 'PAMAP2'

    Returns
    -------
    targetdir: str
        directory where the data is extracted
    """
    targetdir = directory_to_extract_to + '/PAMAP2'
    if os.path.exists(targetdir):
        print('Data previously downloaded and stored in ' + targetdir)
    else:
        os.makedirs(targetdir)  # create target directory
        # Download the PAMAP2 data, this is 688 Mb
        path_to_zip_file = directory_to_extract_to + '/PAMAP2_Dataset.zip'
        test_file_exist = os.path.isfile(path_to_zip_file)
        if test_file_exist is False:
            url = str('https://archive.ics.uci.edu/ml/' +
                      'machine-learning-databases/00231/PAMAP2_Dataset.zip')
            # retrieve data from url
            local_fn, headers = urllib.request.urlretrieve(url,
                                                           filename=path_to_zip_file)
            print('Download complete and stored in: ' + path_to_zip_file)
        else:
            print('The data was previously downloaded and stored in ' +
                  path_to_zip_file)
        # unzip
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(targetdir)
    return targetdir


def slidingwindow_store(y_list, x_list, X_name, y_name, outdatapath, shuffle):
    """
    Take sliding-window frames. Target is label of last time step
    Data is 100 Hz

    Parameters
    ----------
    y_list : list
        list of arrays with classes
    x_list : list
        list of numpy arrays with data
    X_name : str
        Name for X file
    y_name : str
        Name for y file
    outdatapath : str
        directory to store the data
    shuffle : bool
        whether to shuffle the data
    """
    frame_length = int(5.12 * 100)
    step = 1 * 100
    x_set = []
    y_set = []
    sliding_window(frame_length, step, x_set, y_set, x_list, y_list)
    numpify_and_store(x_set, y_set, X_name, y_name,
                      outdatapath, shuffle)


def map_class(datasets_filled):
    ysetall = [set(np.array(data.activityID)) - set([0])
               for data in datasets_filled]
    classlabels = list(set.union(*[set(y) for y in ysetall]))
    nr_classes = len(classlabels)
    mapclasses = {classlabels[i]: i for i in range(len(classlabels))}
    return classlabels, nr_classes, mapclasses


def split_data(Xlists, ybinarylists, indices):
    """ Function takes subset from list given indices

    Parameters
    ----------
    Xlists: tuple
        tuple (samples) of lists (windows) of numpy-arrays (time, variable)
    ybinarylist :
        list (samples) of numpy-arrays (window, class)
    indices :
        indices of the slice of data (samples) to be taken

    Returns
    -------
    x_setlist :
        list (windows across samples) of numpy-arrays (time, variable)
    y_setlist:
        list (windows across samples) of numpy-arrays (class, )
    """
    tty = str(type(indices))
    # or statement in next line is to account for python2 and python3
    # difference
    if tty == "<class 'slice'>" or tty == "<type 'slice'>":
        x_setlist = [X for Xlist in Xlists[indices] for X in Xlist]
        y_setlist = [y for ylist in ybinarylists[indices] for y in ylist]
    else:
        x_setlist = [X for X in Xlists[indices]]
        y_setlist = [y for y in ybinarylists[indices]]
    return x_setlist, y_setlist


def preprocess(targetdir, outdatapath, columns_to_use):
    """ Function to preprocess the PAMAP2 data after it is fetched

    Parameters
    ----------
    targetdir : str
        subdirectory of directory_to_extract_to, targetdir
        is defined by function fetch_data
    outdatapath : str
        a subdirectory of directory_to_extract_to, outdatapath
        is the direcotry where the Numpy output will be stored.
    columns_to_use : list
        list of column names to use

    Returns
    -------
    None
    """
    datadir = targetdir + '/PAMAP2_Dataset/Protocol'
    filenames = listdir(datadir)
    print('Start pre-processing all ' + str(len(filenames)) + ' files...')
    # load the files and put them in a list of pandas dataframes:
    datasets = [pd.read_csv(datadir + '/' + fn, header=None, sep=' ')
                for fn in filenames]
    datasets = addheader(datasets)  # add headers to the datasets
    # Interpolate dataset to get same sample rate between channels
    datasets_filled = [d.interpolate() for d in datasets]
    # Create mapping for class labels
    classlabels, nr_classes, mapclasses = map_class(datasets_filled)
    # Create input (x) and output (y) sets
    xall = [np.array(data[columns_to_use]) for data in datasets_filled]
    yall = [np.array(data.activityID) for data in datasets_filled]
    xylists = [split_activities(y, x) for x, y in zip(xall, yall)]
    Xlists, ylists = zip(*xylists)
    ybinarylists = [transform_y(y, mapclasses, nr_classes) for y in ylists]
    # Split in train, test and val
    x_vallist, y_vallist = split_data(Xlists, ybinarylists, indices=6)
    test_range = slice(7, len(datasets_filled))
    x_testlist, y_testlist = split_data(Xlists, ybinarylists, test_range)
    x_trainlist, y_trainlist = split_data(Xlists, ybinarylists,
                                          indices=slice(0, 6))
    # Take sliding-window frames, target is label of last time step,
    # and store as numpy file
    slidingwindow_store(y_list=y_trainlist, x_list=x_trainlist,
                        X_name='X_train', y_name='y_train',
                        outdatapath=outdatapath, shuffle=True)
    slidingwindow_store(y_list=y_vallist, x_list=x_vallist,
                        X_name='X_val', y_name='y_val',
                        outdatapath=outdatapath, shuffle=False)
    slidingwindow_store(y_list=y_testlist, x_list=x_testlist,
                        X_name='X_test', y_name='y_test',
                        outdatapath=outdatapath, shuffle=False)
    print('Processed data succesfully stored in ' + outdatapath)
    return None


def fetch_and_preprocess(directory_to_extract_to, columns_to_use=None):
    """
    High level function to fetch_and_preprocess the PAMAP2 dataset

    Parameters
    ----------
    directory_to_extract_to : str
        the directory where the data will be stored
    columns_to_use : list
        the columns to use

    Returns
    -------
    outdatapath: The directory in which the numpy files are stored
    """
    if columns_to_use is None:
        columns_to_use = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
                          'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
                          'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z']
    targetdir = fetch_data(directory_to_extract_to)
    outdatapath = targetdir + '/PAMAP2_Dataset/slidingwindow512cleaned/'
    if not os.path.exists(outdatapath):
        os.makedirs(outdatapath)
    if os.path.isfile(outdatapath + 'x_train.npy'):
        print('Data previously pre-processed and np-files saved to ' +
              outdatapath)
    else:
        preprocess(targetdir, outdatapath, columns_to_use)
    return outdatapath


def load_data(outputpath):
    """ Function to load the numpy data as stored in directory
    outputpath.

    Parameters
    ----------
    outputpath : str
        directory where the numpy files are stored
    """
    ext = '.npy'
    x_train = np.load(outputpath + 'X_train' + ext)
    y_train_binary = np.load(outputpath + 'y_train' + ext)
    x_val = np.load(outputpath + 'X_val' + ext)
    y_val_binary = np.load(outputpath + 'y_val' + ext)
    x_test = np.load(outputpath + 'X_test' + ext)
    y_test_binary = np.load(outputpath + 'y_test' + ext)
    return x_train, y_train_binary, x_val, y_val_binary, x_test, y_test_binary
