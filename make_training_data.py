"""
Generate training data for KNC-Live
"""
import argparse
import glob

import pandas as pd

from knc import feature_extraction as fe
from knc import process
from knc.utils import load, ArgumentError

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--force_photo',
                    action='store_true',
                    help='Include forced photometry in lightcurves')
parser.add_argument('--realtime',
                    action='store_true',
                    help='Prepare for data containing recent observations')
parser.add_argument('--data_dir',
                    type=str,
                    help='Directory containing training lightcurves',
                    default='training_data/')
parser.add_argument('--rfc_dir',
                    type=str,
                    help='Directory to store featurized training data',
                    default='classifiers/')
parser.add_argument('--sample',
                    type=int,
                    help='Number of examples to keep in each class',
                    default=None)
parser.add_argument('--verbose',
                    action='store_true',
                    help='Print status updates')

args = parser.parse_args()

# Check command line arguments

# Find simulated lightcurves
lcs_files = glob.glob(f"{args.data_dir}*.npy")

# Extract features and store dfs
feature_dfs = []
for filename in lcs_files:

    # Status update
    if args.verbose:
        print(filename)
    
    # Parse object type from filename
    obj = filename.split('_')[-3]

    # Load lightcurves into memory
    lcs = load(filename)

    # Trim lightcurves to post discovery
    if not args.force_photo:
        lcs = process.trim_lcs(lcs)

    # Extract features
    feat_df = fe.extract_all(
        lcs, cut_requirement=2, obj=obj, sample=args.sample,
        verbose=args.verbose)
    feature_dfs.append(feat_df)

# Determine mode
mode = 'r' if args.realtime else 'f'
if args.force_photo:
    mode += 'fp'
    
# Merge dfs
train_df = pd.concat(feature_dfs)

# Save feature list
with open(f"{args.rfc_dir}features_{mode}.txt", 'w+') as f:
    f.writelines('\n'.join(train_df.columns))
    
# Organize into datasets
if args.verbose:
    print("Organizing")
datasets = process.organize_datasets(train_df)

# Save best featurized dataset
count = 0
best_id = None
for dataset_id, feat_df in datasets.items():
    good_feats = dataset_id.count('F')
    if good_feats > count:
        best_id = dataset_id
        count = good_feats

datasets[best_id].to_csv(f"{args.rfc_dir}training_data_{mode}.csv",
                         index=False)
