from mcfly import tutorial_pamap2
import numpy as np
from nose.tools import assert_equal, assert_equals

import unittest

class TutorialPAMAP2Suite(unittest.TestCase):
    """Basic test cases."""

    def test_addheader(self):
        """
        Tests whether shape of data remains the same
        """
        datasets = dummydataset
        datasetsnew = addheader(datasets)
        test = datasetsnew.shape is datasets.shape
        assert test

if __name__ == '__main__':
    unittest.main()
