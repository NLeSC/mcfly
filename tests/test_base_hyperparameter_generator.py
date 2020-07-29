import numpy as np
from mcfly.models.base_hyperparameter_generator import generate_base_hyperparameter_set, \
    get_learning_rate, get_regularization

def test_regularization_is_float(self):
    """ Regularization should be a float. """
    reg = get_regularization(0, 5)
    assert type(reg) == np.float

def test_regularization_0size_interval(self):
    """ Regularization from zero size interval [2,2] should be 10^-2. """
    reg = get_regularization(2, 2)
    assert reg == 0.01

def test_base_hyper_parameters_reg(self):
    """ Base hyper parameter set should contain regularization. """
    hyper_parameter_set = generate_base_hyper_parameter_set(low_lr=1,
                                                            high_lr=4,
                                                            low_reg=1,
                                                            high_reg=3)

    assert 'regularization_rate' in hyper_parameter_set.keys()
