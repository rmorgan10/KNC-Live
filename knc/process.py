"""
Functionalities to process a lightcurve file for KN-Classify
"""
import argparse
import os

import numpy as np
import pandas as pd

try:
    import feature_extraction
except ModuleNotFoundError:
    import sys
    sys.path.append('knc')
    import feature_extraction

class ArgumentError(Exception):
    """
    A class to raise errors for invalid arguments
    """
    pass

def read_lightcurves(filename : str) -> dict :
    """
    Open a lightcurve file and return the pickeled dictionary

    Args:
        filename (str): Path to lightcurve file

    Returns:
        dictionary containing lightcurves from pickeled file
    """
    return np.load(filename).item()


def organize_datasets(df : pd.DataFrame) -> dict :
    """
    Split a DataFrame of features into separate datasets

    Args:
        df (pandas.DataFrame): DataFrame of lightcurve features

    Returns:
        A dictionary mapping dataset identifiers to the datasets
    """

    def bitrep(arr : list) -> str:
        """
        Convert a boolean array to an inverted string representation

        Args:
            arr (array-like): array of boolean elements

        Returns:
            string of 'T' and 'F' characters with inverted boolean values
        """
        bit_rep = ['F' if val else 'T' for val in arr]
        return ''.join(bit_rep)


    # Organize into datasets with the same sets of well-behaved features
    groups, bad_cols = {}, {}
    for index, row in df.iterrows():
        r = row.copy()
        br = bitrep(row != 'N')

        if br in groups.keys():
            groups[br].append(r.values)
        else:
            groups[br] = [r.values]
            bad_cols[br] = [x for x in df.columns
                            if br[list(df.columns).index(x)] == 'T']

    # Store datasets in a dictionary
    datasets = {k: pd.DataFrame(data=v, columns=df.columns).drop(
        labels=bad_cols[k], axis=1)
                for k, v in groups.iteritems()}

    return datasets
    
def save_datasets(outfile : str, datasets : dict):
    """
    Write datasets to a file

    Args:
        outfile (str): filename to store datasets
        datasets (dict):  A dict mapping dataset identifiers to the datasets 
    """
    np.save(outfile, datasets, allow_pickle=True)


def run_processing(lcs_file : str, results_dir : str = None):
    """
    Run all processing steps on a lightcurve file and save results to disk

    Args:
        lcs_file (str): Path to lightcurve file
        results_dir (str, optional, default=None): directory to save results
    """
    # Establish the results directory
    if results_dir is None:
        results_dir = f'{os.getcwd()}/knc_results'
    if results_dir[-1] == '/':
        results_dir = results_dir[:-1]
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Load data
    lcs = read_lightcurves(lcs_file)

    # Extract features
    feat_df = feature_extraction.extract_all(lcs)

    # Organize into datasets
    datasets = organize_datasets(feat_df)

    # Save results
    save_datasets(f'{results_dir}/KNC_datasets.npy', datasets)


def parse_args() -> argparse.ArgumentParser :
    """
    Parse command line arguments to enable script-like data processing
    
    Returns:
        argparser object
    """

    parser = argparse.ArgumentParser(description=__doc__)

    # Enable command line arguments
    parser.add_argument('--lcs_file',
                        type=str,
                        help='Path to lightcurve file',
                        required=True)
    parser.add_argument('--results_dir',
                        type=str,
                        help='Directory to save results',
                        default=None)

    return parser

def check_args(parser : argparse.ArgumentParser) -> argparse.Namespace :
    """
    Check the arguments for invalid values

    Args:
        parser (argparse.ArgumentParser): a parser object
    
    Returns:
        The parsed arguments if all arguments are valid

    Raises:
        knc.ArgumentError if lcs_file is not passed as argument
        knc.ArgumentError if lcs_file is not found
        knc.ArgumentError if results_dir cannot be found or created
    """

    args = parser.parse_args()

    # Check that the lcs file exists
    if args.lcs_file is None:
        raise ArgumentError("Must pass the lcs_file argument in process mode")
    if not os.path.exists(args.lcs_file):
        raise ArgumentError(f"{args.lcs_file} not found")

    # Check that the results directory can be made or exists
    if args.results_dir is not None:
        if not os.path.exists(args.results_dir):
            
            try:
                os.mkdir(results_dir)
            except FileNotFoundError:
                raise ArgumentError(f"{args.results_dir} is not valid")    

    return args

def process_main(args):
    """
    Run KNC-Live in processing mode

    Args:
        args (argpars.Namespace): parsed arguments for process.py
    """

    # Run data processing
    process.run_processing(args.lcs_file, args.results_dir)

    
if __name__ == "__main__":

    # Get and validate command line arguments
    args = check_args(parse_args())

    # Run data processing
    process_main(args)

