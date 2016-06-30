from mcfly import modelgen
import numpy as np
from nose.tools import assert_equal, assert_equals

import unittest


class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def test_regularization_is_float(self):
        """ Regularization should be a float. """
        reg = modelgen.get_regularization(0, 5)
        assert type(reg) == np.float


    def test_regularization_0size_interval(self):
        """ Regularization from zero size interval [2,2] should be 10^-2. """
        reg = modelgen.get_regularization(2, 2)
        assert_equal(reg, 0.01)


    def test_base_hyper_parameters_reg(self):
        """ Base hyper parameter set should contain regularization. """
        hyper_parameter_set = modelgen.generate_base_hyper_parameter_set()
        assert 'regularization_rate' in hyper_parameter_set.keys()


    def setUp(self):
        np.random.seed(1234)

if __name__ == '__main__':
    unittest.main()