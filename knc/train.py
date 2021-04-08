"""
Train a Random Forest Classifier
"""

import numpy as np
import pandas as pd
pd.set_option('use_inf_as_na', True)
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

try:
    from utils import sigmoid, save
except ModuleNotFoundError:
    import sys
    sys.path.append('knc')
    from utils import sigmoid, save

class Data:
    """
    Organize all data for a ML algorithm
    """
    def __init__(self, df : pd.DataFrame,
                 doit : bool = False,
                 feats: list = []):
        """
        Instantiate a Data object from an input DataFrame. If doit is set to
        True, then the X_train, X_test, y_train, y_test, and feats 
        attributes are calculated
        
        Args:
            df (pd.DataFrame): DataFrame for all objects / features
            doit (bool, optional, default=False): run all data prep steps
            feats (list, optional, default=[]): list of features to consider
        """
        self.data = df

        if doit:
            self.data = self.select_feats(feats)
            self.data = self.clean_data()
            self.prep()
    

    def select_feats(self, feats : list = []) -> pd.DataFrame :
        """
        Select a subset of features for training. Store feats as attribute.

        Args:
            feats (list, optional, default=[]): features to use

        Returns:
            A DataFrame containing only the columns in feats

        Raises:
            ValueError if feats contains names not in columns of self.data
        """
        # Set features to use
        if len(feats) == 0 and not hasattr(self, 'feats'):
            return self.data
        elif len(feats) != 0:
            # Overwrite self.feats if feats are passed to this function
            self.feats = feats
            
        # Check validity of features
        intersection = set(feats).set(self.data.columns)
        if len(intersection) != len(feats):
            raise ValueError("One or more features are not in the data")

        return self.data[feats].copy()

    def clean_data(self) -> pd.DataFrame :
        """
        Remove inf and NaNs from data
            
        Returns:
            df without rows containing infs and NaNs
        """
        # Deal with NaNs and infs
        nas = [np.inf, -np.inf, 'inf', 'nan', 'NaN']
        df = self.data.replace(nas, np.nan).dropna(axis=0, how='any')

        # Force numeric features
        metadata_cols = ['OBJ', 'SNID']
        num_cols = [x for x in df.columns if x not in metadata_cols]
        df[num_cols] = df[num_cols].apply(pd.to_numeric)

        #Drop any extra large values
        df[num_cols] = df[num_cols][~(df[num_cols] > 1.e6).any(axis=1)]

        return df.copy().reset_index(drop=True)

        
    def prep(self):
        """
        Encode and build training and testing sets
        """
        # Apply one-hot encoding
        kn_truth = [1 if x == 'KN' else 0 for x in self.data['OBJ'].values]
        self.data['KN'] = kn_truth

        # Make training and validation sets
        all_feats = [x for x in self.feats if x not in ['SNID', 'CID', 'OBJ']]
        X = self.data[all_feats]
        y = self.data['KN']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=6, stratify=y)

        # Store attributes
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

class Classifier:
    """
    ML algorithm for classification
    """
    def __init__(self, data : Data, doit : bool = False):
        """
        Instantiate a Classifier object. If doit, the best_estimator,
        feature dict, best_params, and feature_importances attirbutes are
        calculated

        Args:
            data (Data) : A prepared instance of the Data class
            doit (bool) : Train a classifier 
        """
        self.data = data
        self.X = data.X
        self.y = data.y
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test
        
        self.rfc = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=6, criterion='gini')

        if doit:
            self.optimize_hyperparamters()
            self.optimize_features()
            self.validate()
            self.fit([], best=True)

    def optimize_hyperparams(self):
        """
        Determine best hyperparamters
        """
        param_grid = {'criterion': ['gini', 'entropy'],
                      'n_estimators': [10, 50, 100, 500],
                      'max_depth': [3, 5, 10, 20],
                      'class_weight': ['balanced_subsample',
                                       'balanced', {0: 1, 1: 1}, {0: 5, 1:5}]}

        gs = GridSearchCV(rfc, param_grid, cv=5)
        gs.fit(self.data.X_train, self.data.y_train)
        self.rfc = gs.best_estimator_
        
        
    def optimize_features(self):
        """
        Determine best features to use
        """
        feature_dict = {}
        feature_names = np.array(self.data.feats)
        fi = self.rfc.feature_importances_
        sorted_fi = sorted(fi)

        # Method 1: use only featrues above maximum gradient
        cut = sorted_fi[np.argmax(np.gradient(sorted_fi))]
        feats = feature_names[np.where(self.rfc.feature_importances_ > cut)]
        if len(feats) > 0:
            rfc_ = self.fit(feats)
            feature_dict[1] = {'FEATURES': feats,
                               'SCORE': rfc.score(self.X_test[feats],
                                                  self.y_test),
                               'CUTOFF': cut}
        else:
            feature_dict[1] = {'FEATURES': feats,
                               'SCORE': 0.0,
                               'CUTOFF': cut}

        # Method 2: use only features above a slightly lower cutoff
        cut = (sorted_fi[np.argmax(np.gradient(sorted_fi))] /
               (0.25 * len(feature_names)))
        feats = feature_names[np.where(rfc.feature_importances_ > cut)]
        if len(feats) > 0:
            rfc = self.fit(feats)
            feature_dict[2] = {'FEATURES': feats,
                               'SCORE': rfc.score(self.X_test[feats],
                                                  self.y_test),
                               'CUTOFF': cut}
        else:
            feature_dict[2] = {'FEATURES': feats,
                               'SCORE': 0.0,
                               'CUTOFF': cut}

        # Method 3: use all features
        rfc = self.fit(feature_names)
        feature_dict[3] = {'FEATURES': feature_names,
                           'SCORE': rfc.score(self.X_test, self.y_test),
                           'CUTOFF': 0.0}

        # Store results
        self.feature_dict = feature_dict

        # Establish best features
        best_score = 0.0
        for info in feature_dict.values():
            if info['SCORE'] > best_score:
                self.feats = info['FEATURES']
                best_score = info['SCORE']
        
            
    def validate(self):
        """
        Evaluate performance on test data and determine calibration
        """
        # Predict on test data
        rfc = self.fit(self.feats)
        scores = rfc.predict_proba(self.X_test)

        # Calculate basic metrics
        precision, recall, pr_thresholds = pr_curve(self.y_test['KN'].values,
                                                    scores)
        f1_score = 2 * (precision * recall) / (precision + recall)
        pr_threshold = pr_thresholds[np.argmax(f1_score)]
        fpr, tpr, roc_thresholds = roc_curve(self.y_test['KN'].values, scores)
        auc = roc_auc_score(self.y_test['KN'].values, scores)
        
        # Determine calibration
        kn_probs, centers = [], []
        for i in range(len(roc_thresholds) - 1):
            
            mask = ((scores >= roc_thresholds[i+1]) &
                    (scores < roc_thresholds[i]))
            if sum(mask) == 0:
                continue
        
            centers.append(0.5 * (roc_thresholds[i] + roc_thresholds[i+1]))
            num_kn = sum(self.y_test['KN'].values[mask] == 1)

            total = sum(mask)
            kn_probs.append(num_kn / total)
            
        popt, pcov = curve_fit(sigmoid, centers, kn_probs)
        self.calibration_coeffs = popt
        self.prob_cutoff = sigmoid(pr_threshold, *popt)

        # Store metrics
        roc_idx = np.argmin(np.abs(roc_thresholds - threshold))
        pr_idx = np.argmin(np.abs(pr_thresholds - threshold))
        self.metrics = {'auc': auc,
                        'fpr': fpr[roc_idx],
                        'tpr': tpr[roc_idx],
                        'precision': precision[pr_idx],
                        'recall': recall[pr_idx],
                        'f1': max(f1_score)}
                                         

        
    def fit(self, feats : list, best : bool = False):
        """
        Fit an optimized RFC with the training data

        Args:
            feats (list): features to use in the fit (ignored if best==True)
            best (bool): Use all data, if false only X_train is used

        Returns:
            a fit RFC if best == False
        """
        if best:
            self.rfc = self.rfc.fit(self.X[self.feats], self.y)
        else:
            return self.rfc.fit(self.X_train[feats], self.y_train)


    def to_dict(self):
        """
        Convert Classifier object to dictionary

        Returns:
            Dictionary where essential attributes of self are the keys
        """
        out_dict = {'rfc': self.rfc,
                    'metrics': self.metrics,
                    'feats': self.feats,
                    'calibration_coeffs': self.calibration_coeffs,
                    'prob_cutoff': self.prob_cutoff,
                    'feature_dict': self.feature_dict}

def train_new(mode : str,
              dataset_id : str,
              key : str,
              rfc_dir : str = 'classifiers/'):
    """
    Train a new classifier and return its key.

    Args:
        mode (str): type of classifier ('r', 'f', 'rfp', 'ffp')
        dataset_id (str): ID string for the dataset
        key (str): ID for the newly trained classifier
        rfc_dir (str, default='classifiers/'): path to classifier directory
    """
    # Load training data
    df = pd.read_csv(f'{rfc_dir}training_data_{mode}.csv')

    # Determine features based on dataset ID
    with open(f"{rfc_dir}features.txt", 'r') as f:
        all_feats = [x for x in f.readlines.split('\n') if x != '']
    feats = [x for i, x in enumerate(all_feats) if dataset_id[i] == 'F'] 

    # Make a Data object
    training_data = Data(df, feats=feats, doit=True)

    # Make a classifier object
    classifier = Classifier(data=training_data, doit=True)
        
    # Save classifier
    save(f"{rfc_dir}knclassifier_{mode}_{key}.npy", classifier.to_dict())
    

    
    
