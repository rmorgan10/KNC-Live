"""
Classify datasets
"""
import argparse

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

def calibrate(scores: np.ndarray) -> np.ndarray :
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

def clean_data(df : pd.DataFrame) -> pd.DataFrame :
    """
    Remove inf and NaNs from data

    Args:
        df (pd.DataFrame): featurized data

    Returns:
        df without rows containing infs and NaNs
    """
    nas = [np.inf, -np.inf, 'inf', 'nan', 'NaN']
    df = df.replace(nas, np.nan).dropna(axis=0, how='any')
    return df.reset_index(drop=True)
    

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


def get_classifier_filename(dataset_id : str,
                            id_map_file : str = 'id_map.npy',
                            rfc_dir : str = 'classifiers/') -> str:
    """
    Given a classifier ID, return the filepath to the classifier

    Args:
        dataset_id (str): ID string for the dataset
        id_map_file (str, default='id_map.npy'): path to map of classifier ids
        rfc_dir (str, default='classifiers/'): path to classifier directory

    Returns:
        filename of the classifier
    """
    if not rfc_dir.endswith('/'):
        rfc_dir += '/'

    id_map = load_classifier(f"{rfc_dir}{id_map_file}")
    key = id_map[dataset_id]

    return f"{rfc_dir}knclassifier_{key}.npy"
    

def classify_datasets(data_dict : dict,
                      id_map_file : str = 'id_map.npy',
                      rfc_dir : str = 'classifiers/') -> pd.DataFrame :
    """
    For each dataset, load the corresponding classifier and predict

    Args:
        data_dict (dict): dictionary containing all datasets
        id_map_file (str, default='id_map.npy'): path to map of classifier ids
        rfc_dir (str, default='classifiers/'): path to classifier directory

    Returns:
        DataFrame with columns SNID and PROB_KN
    """
    classified_data = []
    for dataset_id, df in data_dict.items():
        # Load classifier corresponding to dataset
        try:
            classifier_name = get_classifier_filename(
                dataset_id, id_map_file, rfc_dir)
            
        except KeyError:
            print("WARNING: KNC-Live is not trained for some of the data")
            continue

        # Load classifier
        classifier_dict = load_classifier(classifier_name)
        
        # Remove rows with infs and NaNs
        clean_df = clean_data(df)

        # Apply the classifier
        res = predict(classifier_dict, clean_df)
        classified_data += [(x, y) for x, y in zip(res['SNID'].values,
                                                   res['PROB_KN'].values)]

    return pd.DataFrame(data=classified_data, columns=['SNID', 'PROB_KN'])


def parse_args() -> argparse.ArgumentParser:
    """
    Parse command line arguments to enable script-like classifying

    Returns:
        arrgparser object
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # Enable command line arguments
    parser.add_argument('--datasets_file',
                        type=str,
                        help='Path to datasets file',
                        required=True)
    parser.add_argument('--results_outfile',
                        type=str,
                        help='Filename to store results',
                        default='KNC-Live_Results.csv')
    parser.add_argument('--results_dir',
                        type=str,
                        help='Directory to save results',
                        default=None)
    parser.add_argument('--rfc_dir',
                        type=str,
                        help='Path to directory containing classifiers',
                        default='classifiers/')
    parser.add_argument('--id_map_file',
                        type=str,
                        help='Name of ID map file in classifier directory',
                        default='id_map.npy')

    return parser

def check_args(parser : argparse.ArgumentParser) -> argparse.Namespace :
    """
    Check the arguments for invalid values

    Args:
        parser (argparse.ArgumentParser): a parser object
    
    Returns:
        The parsed arguments if all arguments are valid

    Raises:
        knc.ArgumentError if rfc_dir is not found
        knc.ArgumentError if id_map_file is not found
        knc.ArgumentError if datasets_file is not found
        knc.ArgumentError if results_dir cannot be found or created
    """
    args = parser.parse_args()

    # Check the the classifiers directory exists
    if not os.path.exists(args.rfc_dir):
        raise ArgumentError(f"{args.rfc_dir} not found")
    if not args.rfc_dir.endswith('/'):
        args.rfc_dir += '/'
    
    # Check that the needed files exist
    for filename in [args.datasets_file, args.rfc_dir + args.id_map_file]:
        if not os.path.exists(filename):
            raise ArgumentError(f"{filename} not found")

    # Check that the results directory can be made or exists
    if args.results_dir is not None:
        if not os.path.exists(args.results_dir):
            
            try:
                os.mkdir(results_dir)
            except FileNotFoundError:
                raise ArgumentError(f"{args.results_dir} is not valid")
    else:
        args.results_dir = os.getcwd()

    return args

if __name__ == "__main__":
    
    import argparse
    import os

    from knc.knc import ArgumentError
    from knc.knc import classify_main

    # Get and validate the command line arguments
    args = check_args(parse_args())

    # Run classification and save results
    classify_main(args)
    
