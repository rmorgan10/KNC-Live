"""
Classify datasets
"""
import argparse
import os
import sys
sys.path.append('knc')

import numpy as np
import pandas as pd

import train
from utils import sigmoid, ArgumentError, load, save


def calibrate(scores: np.ndarray, popt : list) -> np.ndarray :
    """
    Transform output scores into probabilities

    Args:
        scores (np.array): scores assigned by classifer
        popt (list): calibration coefficients for sigmoid function

    Returns:
        calibrated scores
    """
    #popt = [2.66766158e-05, 2.66362041e-05, 3.94224999e+01]
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
    rfc = classifier_dict['rfc']
    feats = classifier_dict['feats']
    popt = classifier_dict['calibration_coeffs']
    
    scores = rfc.predict_proba(data[feats])[:,1]
    if not popt is None:
        data['PROB_KN'] = calibrate(scores, popt)
    else:
        data['PROB_KN'] = scores
    data['KN'] = [1 if x >= classifier_dict['prob_cutoff']
                  else 0 for x in data['PROB_KN'].values]

    return data


def get_classifier_filename(mode : str,
                            dataset_id : str,
                            id_map_file : str = 'id_map.npy',
                            rfc_dir : str = 'classifiers/',
                            verbose : bool = False) -> str:
    """
    Given a classifier ID, return the filepath to the classifier. Trains
    a new classifier if no classifiers match the ID.

    Args:
        mode (str): Type of classifier ('r', 'f', 'rfp', 'ffp')
        dataset_id (str): ID string for the dataset
        id_map_file (str, default='id_map.npy'): path to map of classifier ids
        rfc_dir (str, default='classifiers/'): path to classifier directory
        verbose (bool, default=False): Print status updates

    Returns:
        filename of the classifier
    """
    if not rfc_dir.endswith('/'):
        rfc_dir += '/'

    try:
        id_map = load(f"{rfc_dir}{mode}_{id_map_file}")
        key = id_map[dataset_id]
        
    except FileNotFoundError:
        # Initialize the id map and train a classifier
        key = 10000
        id_map = {dataset_id : key}
        if verbose:
            print("No classifier found, training new classifier")
        train.train_new(mode, dataset_id, key, rfc_dir, verbose)
        
    except KeyError:
        if verbose:
            print("No classifier found, training new classifier")
        
        # Train a new classifier and update the id map
        key = max([int(x) for x in id_map.values()]) + 1
        train.train_new(mode, dataset_id, key, rfc_dir, verbose)
        id_map[dataset_id] = key

    # Save the updated id map
    save(f"{rfc_dir}{mode}_{id_map_file}", id_map)

    return f"{rfc_dir}knclassifier_{mode}_{key}.npy"

def classify_datasets(mode : str,
                      data_dict : dict,
                      id_map_file : str = 'id_map.npy',
                      rfc_dir : str = 'classifiers/',
                      verbose : bool = False) -> pd.DataFrame :
    """
    For each dataset, load the corresponding classifier and predict

    Args:
        mode (str): Type of classifier ('r', 'f', 'rfp', 'ffp')  
        data_dict (dict): dictionary containing all datasets
        id_map_file (str, default='id_map.npy'): path to map of classifier ids
        rfc_dir (str, default='classifiers/'): path to classifier directory
        verbose (bool, default=False): Print status updates

    Returns:
        DataFrame with columns SNID and PROB_KN
    """
    out_data = []
    count = 1
    total = len(data_dict)
    for dataset_id, df in data_dict.items():
        # Update status
        if verbose:
            print(f"Classifying dataset {count} of {total}")
            count += 1
        
        # Load classifier corresponding to dataset
        classifier_name = get_classifier_filename(
            mode, dataset_id, id_map_file, rfc_dir, verbose)
            
        # Load classifier
        classifier_dict = load(classifier_name)
        
        # Remove rows with infs and NaNs
        data = train.Data(df)
        clean_df = data.clean_data()

        # Apply the classifier
        res = predict(classifier_dict, clean_df)
        out_data += [(x, y, z) for x, y, z in zip(res['SNID'].values,
                                                  res['PROB_KN'].values,
                                                  res['KN'].values)]

    return pd.DataFrame(data=out_data, columns=['SNID', 'PROB_KN', 'KN'])


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
    parser.add_argument('--mode',
                        type=str,
                        help=('Type of data to classify. r=realtime, f=full, r'
                              'fp=realtime+force_photo, ffp=full+force_photo'),
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
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print status updates')

    return parser

def check_args(parser : argparse.ArgumentParser) -> argparse.Namespace :
    """
    Check the arguments for invalid values

    Args:
        parser (argparse.ArgumentParser): a parser object
    
    Returns:
        The parsed arguments if all arguments are valid

    Raises:
        knc.ArgumentError if mode not in ('r', 'f', 'rfp', 'ffp')
        knc.ArgumentError if rfc_dir is not found
        knc.ArgumentError if id_map_file is not found
        knc.ArgumentError if datasets_file is not found
        knc.ArgumentError if results_dir cannot be found or created
    """
    args = parser.parse_args()

    # Check that the mode is valid
    if not args.mode in ['r', 'f', 'rfp', 'ffp']:
        raise ArgumentError(f"{args.mode} must be r, f, rfp, or ffp")

    # Check that the classifiers directory exists
    if not os.path.exists(args.rfc_dir):
        raise ArgumentError(f"{args.rfc_dir} not found")
    if not args.rfc_dir.endswith('/'):
        args.rfc_dir += '/'
    
    # Check that the processed files exist
    if not os.path.exists(args.datasets_file):
        raise ArgumentError(f"{args.datasets_file} not found")

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

def classify_main(args):
    """
    Run KNC-Live in classification mode

    Args:
        args (argpars.Namespace): parsed arguments for classify.py
    """
    # Run classification
    results = classify_datasets(
        args.mode,
        load(args.datasets_file),
        args.id_map_file,
        args.rfc_dir,
        args.verbose)

    # Save results
    results.to_csv(f"{args.results_dir}{args.results_outfile}", index=False)


if __name__ == "__main__":
    
    # Get and validate the command line arguments
    args = check_args(parse_args())

    # Run classification and save results
    classify_main(args)
    
