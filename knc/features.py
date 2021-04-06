"""
KN-Classify hand-engineered lightcurve features
"""

import numpy as np
import pandas as pd

class FeatureExtractor():
    """
    Class to contain all feature extraction methods
    """
    def __init__(self):
        # Establish feature families
        self.features = [x for x in dir(self) if x[0:1] != '_']
        self.families = ['nobs_brighter_than',
                         'slope',
                         'same_nite_color_diff',
                         'total_color_diff',
                         'snr',
                         'flat',
                         'half',
                         'mag']

        self.feat_families = {fam: [x for x in self.features
                                    if x.find(fam) != -1]
                              for fam in self.families}
        
        self.single_features = [x for x in self.features
                                if x.find('color') == -1]

        self.double_features = [x for x in self.features
                                if x.find('color') != -1]

        return


    # Family 1: Nobs bright than
    def __family1(self, lc, flt, threshold):
        mag_arr = self.__get_mags(lc, flt)
        if len(mag_arr) > 0:
            return sum((mag_arr < threshold))
        else:
            return 'N'

    @staticmethod
    def __get_mags(lc, flt):
        if flt is None:
            return lc['MAG'].values.astype(float)
        else:
            return lc['MAG'].values[lc['FLT'].values == flt].astype(float)

    def nobs_brighter_than_17(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 17.0)
    
    def nobs_brighter_than_18(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 18.0)

    def nobs_brighter_than_19(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 19.0)
    
    def nobs_brighter_than_20(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 20.0)
    
    def nobs_brighter_than_21(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 21.0)

    def nobs_brighter_than_215(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 21.5)
    
    def nobs_brighter_than_22(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 22.0)
    
    def nobs_brighter_than_225(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 22.5)

    def nobs_brighter_than_23(self, lc, flt1, flt2=None):
        return self.__family1(lc, flt1, 23.0)

    def nobs_brighter_than_17_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 17.0)

    def nobs_brighter_than_18_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 18.0)

    def nobs_brighter_than_19_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 19.0)
    
    def nobs_brighter_than_20_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 20.0)

    def nobs_brighter_than_21_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 20.0)

    def nobs_brighter_than_215_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 20.0)

    def nobs_brighter_than_22_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 20.0)

    def nobs_brighter_than_225_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 20.0)

    def nobs_brighter_than_23_any_flt(self, lc, flt1, flt2=None):
        return self.__family1(lc, None, 20.0)


    # Family 2: Slope

    @staticmethod
    def __get_mjds(lc, flt):
        if flt is None:
            mjds = lc['MJD'].values.astype(float)
        else:
            mjds =  lc['MJD'].values[lc['FLT'].values == flt].astype(float)

        if len(mjds) != 0:
            return mjds - mjds.min()
        else:
            return mjds

    def slope_average(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) > 1:
            if mjds[-1] != mjds[0]:
                return (mags[-1] - mags[0]) / (mjds[-1] - mjds[0])
            else:
                return 'N'
        else:
            return 'N'
            
    def slope_max(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) > 1:
            return np.max(np.diff(mags) / np.diff(mjds))
        else:
            return 'N'

    def slope_min(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) > 1:
            return np.min(np.diff(mags) / np.diff(mjds))
        else:
            return 'N'

    def slope_mjd_of_max(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) > 1:
            return mjds[np.argmax(np.diff(mags) / np.diff(mjds))]
        else:
            return 'N'

    def slope_mjd_of_min(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) > 1:
            return mjds[np.argmin(np.diff(mags) / np.diff(mjds))]
        else:
            return 'N'


    # Family 3: Same night color difference
    def __get_nite_color(self, lc, flt1, flt2):
        lc['NITE'] = lc['MJD'].values.astype(float).round().astype(int)
        nites = lc.groupby('NITE')
        colors = []
        for (nite, df) in nites:
            mags_1 = self.__get_mags(df, flt1)
            mags_2 = self.__get_mags(df, flt2)
            if len(mags_1) == 0	or len(mags_2) == 0:
                continue

            colors.append(mags_1.mean() - mags_2.mean())

        return colors

    def same_nite_color_diff_max(self, lc, flt1, flt2):
        colors = self.__get_nite_color(lc, flt1, flt2)
        if len(colors) == 0:
            return 'N'
        else:
            return max(colors)
    
    def same_nite_color_diff_min(self, lc, flt1, flt2):
        colors = self.__get_nite_color(lc, flt1, flt2)
        if len(colors) == 0:
            return 'N'
        else:
            return min(colors)
    
    def same_nite_color_diff_average(self, lc, flt1, flt2):
        colors = self.__get_nite_color(lc, flt1, flt2)
        if len(colors) == 0:
            return 'N'
        else:
            return np.mean(colors)    


    # Family 4: Total color differences
    def total_color_diff_max_max(self, lc, flt1, flt2):
        mags_1 = self.__get_mags(lc, flt1)
        mags_2 = self.__get_mags(lc, flt2)
        if len(mags_1) == 0 or len(mags_2) == 0:
            return 'N'
        else:
            return mags_1.max() - mags_2.max()

    def	total_color_diff_max_min(self, lc, flt1, flt2):
        mags_1 = self.__get_mags(lc, flt1)
        mags_2 = self.__get_mags(lc, flt2)
        if len(mags_1) == 0 or len(mags_2) == 0:
            return 'N'
        else:
            return mags_1.max() - mags_2.min()

    def	total_color_diff_min_max(self, lc, flt1, flt2):
        mags_1 = self.__get_mags(lc, flt1)
        mags_2 = self.__get_mags(lc, flt2)
        if len(mags_1) == 0 or len(mags_2) == 0:
            return 'N'
        else:
            return mags_1.min() - mags_2.max()

    def	total_color_diff_min_min(self, lc, flt1, flt2):
        mags_1 = self.__get_mags(lc, flt1)
        mags_2 = self.__get_mags(lc, flt2)
        if len(mags_1) == 0 or len(mags_2) == 0:
            return 'N'
        else:
            return mags_1.min() - mags_2.min()

    def total_color_diff_mean_mean(self, lc, flt1, flt2):
        mags_1 = self.__get_mags(lc, flt1)
        mags_2 = self.__get_mags(lc, flt2)
        if len(mags_1) == 0 or len(mags_2) == 0:
            return 'N'
        else:
            return mags_1.mean() - mags_2.mean()


    # Family 5: SNR
    @staticmethod
    def __get_flux_and_fluxerr(lc, flt):
        return (lc['FLUXCAL'].values[lc['FLT'].values == flt].astype(float),
                lc['FLUXCALERR'].values[lc['FLT'].values == flt].astype(float))
        
    def snr_max(self, lc, flt1, flt2=None):
        flux, fluxerr = self.__get_flux_and_fluxerr(lc, flt1)
        if len(flux) == 0:
            return 'N'
        else:
            return (flux / fluxerr).max()

    def snr_mean(self, lc, flt1, flt2=None):
        flux, fluxerr =	self.__get_flux_and_fluxerr(lc, flt1)
        if len(flux) ==	0:
            return 'N'
        else:
            return (flux / fluxerr).mean()
        
    def snr_mjd_of_max(self, lc, flt1, flt2=None):
        flux, fluxerr = self.__get_flux_and_fluxerr(lc, flt1)
        if len(flux) == 0:
            return 'N'

        mjds = self.__get_mjds(lc, flt1)
        return mjds[np.argmax(flux / fluxerr)]


    # Family 6: Flat line fitting
    @staticmethod
    def __get_magerrs(lc, flt):
        return lc['MAGERR'].values[lc['FLT'].values == flt].astype(float)

    def flat_reduced_chi2(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 3:
            return 'N'

        chi2 = sum((mags - mags.mean())**2 / magerrs**2)
        dof = len(mags) - 1
        return chi2 / dof

    def flat_reduced_chi2_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 3:
            return 'N'
        
        weighted_av = sum(mags / magerrs**2) / sum(1 / magerrs**2)
        chi2 = sum((mags - weighted_av) ** 2 / magerrs ** 2)
        dof = len(mags) - 1
        return chi2 / dof

    def flat_nobs_3_sigma_from_line(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 3:
            return 'N'
        
        return sum(np.abs((mags - mags.mean()) / magerrs) > 3)

    def flat_nobs_3_sigma_from_line_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 3:
            return 'N'
        
        weighted_av = sum(mags / magerrs**2) / sum(1 / magerrs**2)
        return sum(np.abs((mags - weighted_av) / magerrs) > 3)

    def flat_nobs_2_sigma_from_line(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 3:
            return 'N'

        return sum(np.abs((mags - mags.mean()) / magerrs) > 2)

    def flat_nobs_2_sigma_from_line_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 3:
            return 'N'

        weighted_av = sum(mags / magerrs**2) / sum(1 / magerrs**2)
        return sum(np.abs((mags - weighted_av) / magerrs) > 2)
    

    # Familiy 7: Half lightcurve mags
    
    def half_first_average_mag(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) < 4:
            return 'N'
        split = mjds.mean()
        return mags[mjds < split].mean()

    def half_second_average_mag(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) < 4:
            return 'N'
        split =	mjds.mean()
        return mags[mjds > split].mean()
    
    def half_first_average_mag_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 4:
            return 'N'
        mask = (mjds < mjds.mean())
        return sum(mags[mask] / magerrs[mask]**2) / sum(1 / magerrs[mask]**2)

    def half_second_average_mag_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 4:
            return 'N'
        mask = (mjds > mjds.mean())
        return sum(mags[mask] / magerrs[mask]**2) / sum(1 / magerrs[mask]**2)

    def half_split_average_mag_difference(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        if len(mags) < 4:
            return 'N'
        mask = (mjds < mjds.mean())

        return mags[mask].mean() - mags[~mask].mean() 

    def half_split_average_mag_difference_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        mjds = self.__get_mjds(lc, flt1)
        magerrs = self.__get_magerrs(lc, flt1)
        if len(mags) < 4:
            return 'N'
        mask = (mjds < mjds.mean())

        lav = sum(mags[mask] / magerrs[mask]**2) / sum(1 / magerrs[mask]**2) 
        rav = sum(mags[~mask] / magerrs[~mask]**2) / sum(1 / magerrs[~mask]**2)
        return lav - rav


    # Family 8: Full lightcurve mags
    def mag_average(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        if len(mags) == 0:
            return 'N'
        return mags.mean()

    def mag_average_weighted(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        magerrs = self.__get_mags(lc, flt1)
        if len(mags) == 0:
            return 'N'
        return sum(mags / magerrs**2) / sum(1 / magerrs**2)

    def mag_brightest(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        if len(mags) == 0:
            return 'N'
        return mags.min()

    def mag_total_change(self, lc, flt1, flt2=None):
        mags = self.__get_mags(lc, flt1)
        if len(mags) == 0:
            return 'N'
        return mags[0] - mags[-1] 

