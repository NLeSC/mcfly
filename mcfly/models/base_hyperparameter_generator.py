import numpy as np

def generate_base_hyperparameter_set(low_lr, high_lr, low_reg, high_reg):
    """Generate a base set of hyperparameters that are necessary for any
    model, but sufficient for none.

    Parameters
    ----------
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`

    Returns
    -------
    hyperparameters : dict
        basis hyperpameters
    """
    hyperparameters = {}
    hyperparameters['learning_rate'] = get_learning_rate(low_lr, high_lr)
    hyperparameters['regularization_rate'] = get_regularization(
        low_reg, high_reg)
    return hyperparameters


def get_learning_rate(low, high):
    """Return random learning rate 10^-n where n is sampled uniformly between
    low and high bounds.

    Parameters
    ----------
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    learning_rate : float
        learning rate
    """
    result = 10 ** (-np.random.uniform(low, high))
    return result


def get_regularization(low, high):
    """Return random regularization rate 10^-n where n is sampled uniformly
    between low and high bounds.

    Parameters
    ----------
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    regularization_rate : float
        regularization rate
    """
    return 10 ** (-np.random.uniform(low, high))
