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
