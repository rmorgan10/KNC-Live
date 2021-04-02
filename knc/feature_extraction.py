"""
Extract features from lightcurves
"""

import pandas as pd

from knc.features import FeatureExtractor


def extract(lc : pd.DataFrame,
            extractor : FeatureExtractor,
            flts : str ="griz") -> dict :
    """
    Extract features from a single lightcurve

    Args:
        lc (pandas.DataFrame): A single lightcurve
        extractor (FeatureExtractor): A features.FeatureExtractor instance
        flts (str or list, default='griz'): iterable of all possible filters
    
    Returns:
        dict of feature names and values
    """

    # Determine which filters are present in the lightcurve
    good_flts = list(set(lc['FLT'].values))
    data_dict = {}
    for flt in flts:
        if flt in good_flts:
            flt_good = True
        else:
            flt_good = False

        # Extract all single-filter features
        for feat in extractor.single_features:
            if flt_good:
                command = f'extractor.{feat}(lc, flt)'
                data_dict[feat + '_' + flt] = eval(command)
            else:
                data_dict[feat + '_' + flt] = 'N'

    # Extract all double-filter features
    for pair in ['gr', 'gi', 'gz', 'ri', 'rz', 'iz']:
        for feat in extractor.double_features:
            command = f'extractor.{feat}(lc, pair[0], pair[1])'
            data_dict[feat + '_' + pair] = eval(command)

    return data_dict

def extract_all(lcs : dict,
                cut_requirement: int = 0,
                obj : str = 'DATA') -> pd.DataFrame:
    """
    Extract features from all lightcurves in a dictionary

    Args:
        lcs (dict): dict of all lightcurves
        cut_requirement (int, default=0): number of cut to enforce
        obj (str, default='DATA') : label to give to object
    
    Returns:
        pandas DataFrame of all extracted features
    """
    extractor = FeatureExtractor()

    # Extract features for everything at the desired cut level
    data = []
    for snid, info in cut_res.items():
        if info['cut'] > cut_requirement or info['cut'] == -1:
            flts = set(info['lightcurve']['FLT'].values)
            data_dict = extract(info['lightcurve'], flts, extractor)
            data_dict['SNID'] = snid
            data_dict['OBJ'] = obj
            data.append(data_dict)

    # Construct and clean a DataFrame
    df = pd.DataFrame(data)
    df = df.dropna(how='all')
    df = df.fillna('N')

    return df
