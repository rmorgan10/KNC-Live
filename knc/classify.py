"""
Classify datasets
"""

import numpy as np
import pandas as pd

def load_classifier(filename : str) -> dict :
    """
    Load a classifier from a pickled file

    Args:
        filename (str): name of classifer file

    Returns:
        dictionary of classifier and features to use
    """
    return np.load(filename, allow_pickle=True).item()

def calibrate(scores: np.array) -> np.array :
    """
    Transform output scores into probabilities

    Args:
        scores (np.array): scores assigned by classifer
 
    Returns:
        calibrated scores
    """
    def sigmoid(x, a, b, c):
        return a / (b + np.exp(-1.0 * c * x))
    
    popt = [2.66766158e-05, 2.66362041e-05, 3.94224999e+01]
    return sigmoid(scores, *popt)
    

def predict(classifier_dict : dict, data : pd.DataFrame) -> pd.DataFrame :
    """
    Predict on data and add results to the dataframe

    Args:
        classifier_dict (dict): dictionary of classifier and features to use 
        data (pd.DataFrame): featurized dataset examples

    Returns:
        DataFrame with a 'PROB_KN' column
    """
    rfc = classifier_dict['classifier']
    feats = classifier_dict['feats']

    scores = rfc.predict_proba(data[feats])[:,1]
    data['PROB_KN'] = calibrate(scores)

    return data

