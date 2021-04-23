#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:31:52 2021

@author: Mitchell
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot

from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

#------tests-----------
#test for ADF - null hypothesis that a unit root is present in a time series sample
#- null hypothesis that a unit root is present in a time series sample
def test_adfuller(arr,to_print = False):
    X1 = np.array(arr)
    X1 = X1[~np.isnan(X1)]
    
    result = adfuller(X1)
    if to_print:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        
    adf_stat, p_val = result[0], result[1]
    return adf_stat, p_val

#kpss test
def kpss_test(series, to_print = False, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    if to_print:
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critial Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')
        print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    return p_value

#-----------------------
def check_stationarity(df):
    for i in range(df.shape[1]):
        ts = df.iloc[:,i].dropna()
        name = df.columns[i]
#       adf test and kpss test
        adf_stat, p_val = test_adfuller(ts)
        # kpss_p = kpss_test(ts)
        kpss_p = 1000
        print(name, p_val)
        
        # print(p_val, kpss_p)
        if p_val >= 0.05 or kpss_p < 0.05:
            print('{} non stationary due to ADF and KPSS'.format(name))
        elif p_val >= 0.05:
            print('{} non stationary due to ADF'.format(name))
        elif kpss_p < 0.05:
            print('{} non stationary due to KPSS'.format(name))
            
            
            
                
def fit_distribution(dataframe):
    dist_names = ['weibull_min','norm','weibull_max','beta','invgauss',
                  'uniform','gamma','expon','lognorm','pearson3','triang']
    
    
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data 
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)
    
    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        print("{}\n{}\n".format(dist, param))
    
    
        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
    
        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)
    
    
    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)
    
    
    print ('\nDistributions listed by Betterment of fit:')
    print ('............................................')
    print (results)
    