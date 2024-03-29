# KN-Classify Live

A live version of KN-Classify for DESI-DOT data.

Written by Robert Morgan.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4681075.svg)](https://doi.org/10.5281/zenodo.4681075)

```
@software{robert_morgan_2021_4681075,
  author       = {Robert Morgan},
  title        = {KN-Classify},
  month        = apr,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.4681075},
  url          = {https://doi.org/10.5281/zenodo.4681075}
}
```

## Environment Setup

Other environments with more recent versions of python and scikit-learn will probably work too, but they have not been tested.

`$ conda create -n knc_env python=3.7.1 scikit-learn=0.22.2.post1 numpy pandas scipy matplotlib jupyter`

`$ conda activate knc_env`

## Installation

Just clone the repository.

`git clone https://github.com/rmorgan10/KNC-Live.git`

## Usage

### Step 0: Format your lightcurves

Store your lightcurves in a dictionary with this structure:

```python
lcs = {'<SNID 1>': {'lightcurve': <DataFrame 1>, 'cut': <value from snr5>},
       '<SNID 2>': {'lightcurve': <DataFrame 2>, 'cut': <value from snr5>},
       ...}
```

The required columns in the `DataFrame`s are `MJD`, `FLT`, `MAG`, `MAGERR`, `FLUXCAL`, and `FLUXCALERR`, where each row is an observation of the transient object.

The value for `cut` should be the output of this function:

```python
def snr5(lc : pd.DataFrame) -> int:
    """ Return -1 if lc passes a SNR cut, 1 otherwise """
    flux = lc['FLUXCAL'].values.astype(float)
    fluxerr = lc['FLUXCALERR'].values.astype(float)
    if (flux / fluxerr).max() >= 5.0:
        return -1
    else:
        return 1
```

Lastly, save the lightcurves:

```python
np.save("<outfile_name>.npy", lcs, allow_pickle=True)
```

### Step 1: Copy your formatted lightcurves to the `testing_data` directory

You can technically use any directory on your system, but for sumplicity, the `testing_data` directory is included in the repo. Put any formatted lightcurve files that you want to classify in that directory.

### Step 2: Run KNC-Live

The `run_knc.py` script is included to help utilize the main features of KN-Classify. You can see what the arguments for this script are with the `--help` argument:

```python
$ python run_knc.py --help
usage: run_knc.py [-h] [--process] [--classify] [--lcs_file LCS_FILE]
                  [--datasets_file DATASETS_FILE] [--mode MODE]
                  [--results_outfile RESULTS_OUTFILE]
                  [--results_dir RESULTS_DIR] [--rfc_dir RFC_DIR]
                  [--id_map_file ID_MAP_FILE] [--verbose] [--skip_cv]
                  [--distribute]

Run KN-Classify Live

optional arguments:
  -h, --help            show this help message and exit
  --process             Run data processing
  --classify            Run classification
  --lcs_file LCS_FILE   Path to input lcs data file
  --datasets_file DATASETS_FILE
                        Path to datasets file or output of processing
  --mode MODE           Type of data to classify. r=realtime, f=full,
                        rfp=realtime+force_photo, ffp=full+force_photo
  --results_outfile RESULTS_OUTFILE
                        Filename to store results
  --results_dir RESULTS_DIR
                        Directory to save results
  --rfc_dir RFC_DIR     Path to directory containing classifiers
  --verbose             Print status updates
  --skip_cv             Skip hyperparam optimization
  --distribute          Use multiprocessing

```

**The `mode` Argument**:
This is the most important argument, becasue using it incorrectly will produce inaccurate classifications.
There are four options: 

| mode | Meaning |
| --- | --- |
| r | _realtime_: The data contain 1 to 8 observing epochs, with no photometry before the transient detection |
| rfp | _realtime+forcephoto_: The data contain 1 to 8 observing epochs with forced photometry before the transient detection |
| f | _full_: The data contain all the epochs from the DESI-DOT program, with no photometry before the transient detection |
|ffp | _full+forcephoto_: The data contain all the epochs from the DESI-DOT program with forced photometry before the transient detection |

### Example

```python
python run_knc.py --process --classify --verbose --lcs_file testing_data/20210324_2Dets.npy --mode r --datasets_file testing_data/real_datasets.npy --results_dir knc_results/ --rfc_dir classifiers/
```

This command will find your `realtime`-style formatted lightcurves in `testing_data/20210324_2Dets.npy`, process it and save the featurized data in `testing_data/real_datasets.npy`, then classify the featurized data using the training data and classifiers in `classifiers`, and lastly store the final output in `knc_results/`. It will also print status updates as it chugs along.

### Speed Considerations

The processing part of KN-Classify scales linearly with the number of objects to process. 
It will be sped up in future versions, but for now you can process the data once and then use the saved results every time you want to classify your data.

The classification part of KN-Classify is quite fast, but will take a significant amount of time to train whenever a new type of dataset is detected.
To speed up the training you have two options: `--skip_cv` and `--distribute`

- passing `--skip_cv` to run_knc.py or classify.py will skip the hyperparameter optimization step of the training process and use a default classifier. You will likely still achieve an accuracy above 95%, but a fully optimized classifier will not be trained.
- passing `--distribute` to run_knc.py or classify.py will utilize all available cores to fit classifiers. This setting is recommended if you are working locally. If you are working on a login node of a cluster, you may risk your job getting killed for using too many resources.

## API Documentation

https://rmorgan10.github.io/KNC-Live/docs/
