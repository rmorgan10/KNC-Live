"""
Utility functions for KNC-Live
"""

import numpy as np

def sigmoid(x : np.ndarray, a : float, b : float, c : float) -> np.ndarray :
    """
    A parameterized sigmoid curve

    Args:
        x (np.ndarray or float): x values to evaluate the sigmoid
        a (float): vertical stretch parameter
        b (float): horizontal shift parameter
        c (float): horizontal stretch parameter

    Returns:
        evalutated sigmoid curve at x values for the given parameterization
    """
    return a / (b + np.exp(-1.0 * c * x))

def load(filename : str) -> object:
    """
    Load a pickled file into memory. Warning: Only use on data you trust 
    because this function overrides the default pickling allowance in numpy,
    which is there for security reasons.

    Args:
        filename (str): path to file containing pickled object

    Returns:
        unpickled object
    """
    return np.load(filename, allow_pickle=True).item()

def save(filename : str, obj : object):
    """
    Save an object by pickling

    Args:
        filename (str): path to file that will contain pickled object
        obj (object): a python object you want to pickle
    """
    np.save(filename, obj, allow_pickle=True)

class ArgumentError(Exception):
    """
    A class to raise errors for invalid arguments
    """
    pass
