from mcfly import tutorial_pamap2
import numpy as np
from nose.tools import assert_equal, assert_equals

import unittest

class TutorialPAMAP2Suite(unittest.TestCase):
    """Basic test cases."""

    def test_addheader(self):
        """ Test whether function produces dataframe of same shape as input
        """
        datasets = pd.DataFrame(index=range(100),columns=range(54))
        datasetsnew = addheader(datasets)
        test = datasetsnew.shape is datasets.shape
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
        if test is True:
            os.remove(filename)
        assert test


if __name__ == '__main__':
    unittest.main()
