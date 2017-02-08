from tutorial import tutorial_pamap2
import numpy as np
import pandas as pd
from nose.tools import assert_equal, assert_equals
from os import listdir
import os.path
import unittest


class TutorialPAMAP2Suite(unittest.TestCase):
    """Basic test cases."""

    def test_split_activities(self):
        """
        Test whether split_activities produces a Numpy array
        """
        labels = np.ones(3000)
        labels[range(150)] = 2
        X = np.ones((3000,9))
        splittedX = tutorial_pamap2.split_activities(labels,X,[0])
        test = splittedX[0][0].shape == (1150, 9)
        assert test

    def test_sliding_window(self):
        """ Test whether sliding_window correctly updates x_train to the
         right size"""
        frame_length = 512
        step = 100
        x_trainlist = [np.zeros((25187,9)) for b in range(78)]
        y_trainlist = [np.zeros((12,9)) for b in range(78)]
        x_train, y_train = tutorial_pamap2.sliding_window(frame_length, step, x_trainlist, y_trainlist)
        test = len(x_train) == 19266
        assert test

    def test_transform_y(self):
        """ Test whether function produces Numpy array of expected size """
        mapclasses = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, \
                        12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
        nr_classes = 12
        y = list([1,2,5,7,13,16,24,1,2,5,7,13,16,24]) #14 values
        transformedy = tutorial_pamap2.transform_y(y, mapclasses, nr_classes)
        test = transformedy.shape == (14,12)
        assert test

    def test_addheader(self):
        """ Test whether addheader produces dataframe of same shape as input
        """
        datasets = [pd.DataFrame(index=range(100),columns=range(54)) for b in range(10)]
        datasetsnew = tutorial_pamap2.addheader(datasets)
        test = datasetsnew[0].shape == datasets[0].shape
        assert test

    def test_numpify_and_store(self):
        """ Test whether numpify_and_store produces npy-file """
        Nsamples = 9
        Ntimesteps = 10
        Ncolumns = 3
        X = [[[0 for a in range(Ncolumns)] for b in range(Ntimesteps)] \
            for c in range(Nsamples)]
        y = [[0 for a in range(Ntimesteps)] for b in range(Nsamples)]
        xname = 'xname'
        yname = 'yname'
        outdatapath = os.getcwd()
        tutorial_pamap2.numpify_and_store(X, y, xname, yname, outdatapath, \
            shuffle=True)
        filename = os.path.join(outdatapath, xname+ '.npy')
        test = os.path.isfile(filename)
        if test == True:
            os.remove(filename)
            os.remove(os.path.join(outdatapath, yname + '.npy'))
        assert test

    def test_split_data(self):
        """ Test whether function produces numpy arrays
        of the correct dimensions """
        Xlists = tuple([[np.zeros((200,9)) for b in range(14)] for c in range(9)])
        ybinarylists = [np.zeros((14,12)) for c in range(9)]
        indices = slice(7, 9)
        x_test, y_test = tutorial_pamap2.split_data(Xlists, ybinarylists, \
            indices)
        test = y_test[0].shape == (12,) and x_test[0].shape == (200, 9)
        assert test


if __name__ == '__main__':
    unittest.main()
