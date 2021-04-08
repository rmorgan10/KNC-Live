"""
Run KN-Classify Live
"""

import os
import argparse

from knc import process
from knc import classify
from knc.utils import ArgumentError


def parse_args() -> argparse.ArgumentParser:
    """
    Parse command line arguments to enable script-like running of KNC-Live

    Returns:
        arrgparser object
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # Enable command line arguments
    parser.add_argument('--process',
                        action='store_true',
                        help='Run data processing')
    parser.add_argument('--classify',
                        action='store_true',
                        help='Run classification')
    parser.add_argument('--lcs_file',
                        type=str,
                        help='Path to lcs file')
    parser.add_argument('--datasets_file',
                        type=str,
                        help='Path to datasets file')
    parser.add_argument('--mode',
                        type=str,
                        help=('Type of data to classify. r=realtime, f=full, r'
                              'fp=realtime+force_photo, ffp=full+force_photo'))
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

if __name__ == '__main__':

    # Get arguments
    parser = parse_args()
    args = parser.parse_args()

    # Validate arguments
    if args.process:
        process_args = process.check_args(parser)

    if args.classify:
        classify_args = classify.check_args(parser)

    # Run scripts
    if args.process:
        process.process_main(process_args)

    if args.classify:
        classify.classify_main(classify_args)
