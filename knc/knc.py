"""
Run KN-Classify Live
"""

import os
import argparse

import knc.process 
import knc.classify 

class ArgumentError(Exception):
    """
    A class to raise errors for invalid arguments
    """
    pass

def classify_main(args):
    """
    Run KNC-Live in classification mode

    Args:
        args (argpars.Namespace): parsed arguments for classify.py
    """
    # Run classification
    results = knc.classify.classify_datasets(
        load_classifier(args.datasets_file),
        args.id_map_file,
        args.rfc_dir)

    # Save results
    results.to_csv(f"{args.results_dir}{args.results_outfile}", index=False)
    

def process_main(args):
    """
    Run KNC-Live in processing mode

    Args:
        args (argpars.Namespace): parsed arguments for process.py
    """

    # Run data processing
    knc.process.run_processing(args.lcs_file, args.results_dir)


def parse_args() -> argparse.ArgumentParser:
    """
    Parse command line arguments to enable script-like running of KNC-Live

    Returns:
        arrgparser object
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # Enable command line arguments
    parser.add_argument('--process',
                        action='store True',
                        help='Run data processing')
    parser.add_argument('--classify',
                        action='store True',
                        help='Run classification')
    parser.add_argument('--lcs_file',
                        type=str,
                        help='Path to lcs file')
    parser.add_argument('--datasets_file',
                        type=str,
                        help='Path to datasets file')
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

if __name__ == '__main__':

    # Get arguments
    parser = parse_args()
    args = parser.parse_args()

    # Validate arguments
    if args.process:
        process_args = knc.process.check_args(parser)

    if args.classify:
        classify_args = knc.classify.check_args(parser)

    # Run scripts
    if args.process:
        process_main(process_args)

    if args.classify:
        classify_main(classify_args)
