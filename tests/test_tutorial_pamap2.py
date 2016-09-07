from mcfly import tutorial_pamap2
import numpy as np
from nose.tools import assert_equal, assert_equals

import unittest

class TutorialPAMAP2Suite(unittest.TestCase):
    """Basic test cases."""


    def test_split_activities(self):
        """
        Test whether function produces a Numpy array
        """
        labels = np.ones(3000)
        labels[range(150)] = 2
        X = np.ones((3000,9))
        splittedX = split_activities(labels,X)
        test = splittedX[0][0].shape == (1150, 9)
        assert test

    def test_sliding_window(self):
        """ Test whether function correctly updates x_train to the
         right size"""
        frame_length = 512
        step = 100
        x_trainlist = [np.zeros((25187,9)) for b in range(78)]
        y_trainlist = [np.zeros((12,9)) for b in range(78)]
        x_train = []
        y_train = []
        sliding_window(frame_length, step, x_train, y_train, y_trainlist, \
            x_trainlist)
        test = len(x_train) == 19266
        assert test

    def test_transform_y(self):
        """ Test whether function produces Numpy array of expected size """
        mapclasses = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, \
                        12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
        nr_classes = 12
        y = list([1,2,5,7,13,16,24,1,2,5,7,13,16,24]) #14 values
        transformedy = transform_y(y, mapclasses, nr_classes)
        test = transformedy.shape == (14,12)
        assert test

    def test_addheader(self):
        """ Test whether function produces dataframe of same shape as input
        """
        datasets = pd.DataFrame(index=range(100),columns=range(54))
        datasetsnew = addheader(datasets)
        test = datasetsnew.shape == datasets.shape
        assert test

    def test_numpify_and_store(self):
        """ Test whether function produces npy-file """
        Nsamples = 9
        Ntimesteps = 10
        Ncolumns = 3
        X = [[[0 for a in range(Ncolumns)] for b in range(Ntimesteps)] \
            for c in range(Nsamples)]
        y = [[0 for a in range(Ntimesteps)] for b in range(Nsamples)]
        xname = 'xname'
        yname = 'yname'
        outputpath = os.getcwd()
        numpify_and_store(X, y, xname, yname, outdatapath, shuffle=True)
        filename = outdatapath+ xname+ '.npy'
        test = os.path.isfile(filename)
        if test == True:
            os.remove(filename)
        assert test


if __name__ == '__main__':
    unittest.main()
