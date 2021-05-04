#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:07:34 2021

@author: mitchell
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline     
## use `%matplotlib notebook` for interactive figures
#plt.style.use('ggplot')
import sklearn
from tigramitecustom import data_processing as pp
from tigramitecustom import plotting as tp
from tigramitecustom.pcmci import PCMCI
from tigramitecustom.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramitecustom.models import LinearMediation, Prediction

from tigramitecausationsupdated import  subtract_rolling_mean, take_first_diff, adjust_seasonality, create_data

from itertools import chain
from extractdata import extract_giews, get_attribute
from eeDataExtract import make_enviro_data



#
#commodity = 'Rice'
#study_market = 'Dakar'
#add_enviro  = True
#
##whether or not to restrict linear regression weights to only positive values. 
##If so, environmental variables are multiplied by -1.
#restrict_positive = True
#
#condition_on_my = True
#
#interpolate = True
#inter_max_gap = 3
#
##allows for saving enviro data over multiple runs
#if 'enviro_data_dict' not in dir():
#    enviro_data_dict = {} 
#
#
#s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,2,28)
###------------import rice------------
##mkts_dict, fao_mkts_dict = get_rice_dict()
##
#minimum_size = 160
#
#max_lag = 4
#min_lag = 1 #(tau_min)


#rice_dataframe = get_rice_df(fao_mkts_dict, None, 160, s, e)
#
## -----------import millet---------
##senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
##millet_dataframe = GEIWS_prices(senegal_millet_file)
#
#millet_prices = pd.DataFrame.from_dict(extract_giews(country = 'Senegal', commodity = 'Millet', min_size = minimum_size))


def run_pred_test(country, commodity, study_market, steps_ahead,  tau_max, add_enviro, study_variables = None,  m_y_conditioning = True, 
             interpolate = False, max_gap = 3, print_info = False,  use_gee = True, minimum_size = 160):
    ''' 
    Code to run predictive test based on tigramite PCMCI network framework. 
    Optimized for use in Senegal with Rice and Millet but applicable to any country/commodity combination
    '''
    
    
    data = create_data(country, commodity,min_size = minimum_size)
    #  set up environmental dataframe if valid   
    if add_enviro: # make dataframe for NDVI and Precip over regions of commodity growth, 
                   # flip to negative if restricting lin regression to positive       
        if commodity in enviro_data_dict.keys():
            enviro_df = enviro_data_dict[commodity]
        else:
            if use_gee:
                enviro_df =   make_enviro_data(commodity) 
            else:
                enviro_df = pd.read_csv('envirodata/{}-fullenviro.csv'.format(commodity.lower()) )
                enviro_df.index = pd.to_datetime(enviro_df.iloc[:,0], format = '%Y-%m-%d')
                enviro_df =  enviro_df.drop(columns = enviro_df.columns[0])
            enviro_data_dict[commodity] = enviro_df
        enviro_df = - enviro_df if restrict_positive == True else enviro_df
        
        
    # define target index
    target = list(data.columns).index(study_market)
    
    #get stats for time series for later correction
    mean = np.nanmean(data.iloc[:,target])
    std = np.nanstd(data.iloc[:,target])
    
    #define function to "denormalize" data back to real values
    def denorm(z):
        return (z*std) + mean
    

    
    #study_vars = millet_dataframe.columns
    input_str = 'Millet - All markets and NDVI/Precip in Kaolack, Kaffrine, and Fatick' if commodity == 'millet' else 'Rice - Bangkok, Sao Paolo, Mumbai, and NDVI/Precip in SRV and Casamance'
    
    mssng = 99999
    #data_clipped = data[s:e]
    
    #data_stationary = subtract_rolling_mean( adjust_seasonality( data.copy()))[s:e]

    
    # if conditioning on month year
    if condition_on_my:
        m_y_data = data.copy()[s:e]
    #        if adding environmental variables
        if add_enviro:
            m_y_data = pd.concat([m_y_data, enviro_df], axis = 1)
            enviro_indices = [m_y_data.columns.get_loc(x) for x in enviro_df.columns ]
            
        m_y_data['Month'] = m_y_data.index.month
        m_y_data['Year'] = m_y_data.index.year
        m_y_indices = [m_y_data.columns.get_loc('Month'), m_y_data.columns.get_loc('Year')]
        m_y_data = m_y_data.interpolate(method='linear', limit=inter_max_gap) if interpolate == True else  m_y_data
        data_filled = m_y_data.fillna(mssng)
        
    else:
        pass
        
        
    data_filled = data_filled[study_variables] if study_variables == True else data_filled   
    
    T, N = data_filled.shape
    
    selected_links = {i : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] 
                                    for j in range(N) if j != target ])) for i in range(N)}
    
    dataframe = pp.DataFrame(data_filled.values, var_names = data_filled.columns, missing_flag = mssng)
    
    #links to study
    selected_links = {i : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] 
                                    for j in range(N) if j != target ])) for i in range(N)}
    #    remove month and year conditions
        
    
    
    
    #data_stationary_filled = data_stationary.fillna(mssng)
    #data_filled = data_filled.dropna()
    #data_stationary_filled = data_stationary_filled[study_vars] if use_study_vars == True else data_stationary_filled 
    
    
    
    #dataframe = pp.DataFrame(data_filled.values, var_names = data.columns)
    #dataframe_stationary = pp.DataFrame(data_stationary_filled.values, var_names = data_stationary_filled.columns, missing_flag = mssng)
    
    #-------- Goal 1 -----------
    #- Predict Dakar with international and environmental time series
    
    
    # We initialize the Prediction class with cond_ind_model=ParCorr(). 
    # Secondly, we choose sklearn.linear_model.LinearRegression() here for prediction. Last, we scale the
    #  data via data_transform. The class takes care of rescaling the data for prediction. The parameters 
    #  train_indices and test_indices are used to divide the data up into a training set and test set. 
    #  The test set is optional since new data can be supplied later. The training set is used to select 
    #  predictors and fit the model
    
    train_indices = range(int(0.8*T))
    test_indices = range(int(0.8*T), T)
    
    pred = Prediction(dataframe=dataframe,
            cond_ind_test=ParCorr(),   #CMIknn ParCorr
            prediction_model = sklearn.linear_model.LinearRegression(positive = restrict_positive),
    #         prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
    #         prediction_model = sklearn.linear_model.BayesianRidge(),
    #         prediction_model = sklearn.neighbors.KNeighborsRegressor(),
             
        data_transform=sklearn.preprocessing.StandardScaler(),
        train_indices= train_indices,
        test_indices= test_indices,
        verbosity=1, 
        print_info = print_info
        )
    
    #pred_stationary = Prediction(dataframe= dataframe_stationary,
    #        cond_ind_test=ParCorr(),   #CMIknn ParCorr
    #        prediction_model = sklearn.linear_model.LinearRegression(positive = restrict_positive),
    #    data_transform=sklearn.preprocessing.StandardScaler(),
    #    train_indices= train_indices,
    #    test_indices= test_indices,
    #    verbosity=1
    #    )
    # Now, we estimate causal predictors using get_predictors for the target variable 2 taking into account a maximum 
    # past lag of tau_max. We use pc_alpha=None which optimizes the parameter based on the Akaike score. Note that 
    # the predictors are different for each prediction horizon. For example, at a prediction horizon of steps_ahead=1 
    # we get the causal parents from the model plus some others:
    
    
    
    
    #studied_link_vars = 
    selected_links = {i : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] 
                                    for j in range(N) if j != target ])) for i in range(N)}
    #selected_links = {target : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] for j in range(pred.N) ])) }
        
        
    predictors = pred.get_predictors(
                      selected_targets=[target],
                      selected_links = selected_links,
                      steps_ahead=steps_ahead,
                      tau_max=tau_max,
                      pc_alpha=None
                      )
    
    
    print('\n\n#### Target Value Predictors: #### \n')
    print(*[str(data_filled.columns[tup[0]]) +', lag = '+ str(tup[1]) for tup in predictors[target]], sep = "\n")
    link_matrix = np.zeros((N, N, tau_max+1), dtype='bool')
    for j in [target]:
        for p in predictors[j]:
            link_matrix[p[0], j, abs(p[1])] = 1
            
    # Plot time series graph
    if print_info:
        tp.plot_time_series_graph(
            figsize=(6, 3),
        #     node_aspect=2.,
            val_matrix=np.ones(link_matrix.shape),
            link_matrix=link_matrix,
            var_names=None,
            link_colorbar_label='',
            ); plt.show()
    
    #------
    
    pred.fit(target_predictors= predictors, 
                    selected_targets=[target],
                        tau_max=tau_max,
                        return_data = True)
    
    
    
    new_data_vals = data_filled[study_variables].copy() if study_variables == True else data_filled.copy()
    new_data_vals = new_data_vals.replace(mssng,np.nan)
    
    #-----adjustment for each variable --------
    # dict of form:   variable : [ z_shift (float),  steps back (int, how far back to apply shock) ]
    # z_shift is how many z-scores to shift data
    steps_back = 12
    adjustment_params = { var_name : [0, steps_back] for var_name in new_data_vals.columns}
    
    
    #data is adjusted here:
    
    #------ increase environmental time series by half a z-score --------
    #rice
    if commodity.lower() == 'rice':
    #    adjustment_params['NorthernRiverValley_NDVI'][0] = 0.5
    #    adjustment_params['SouhternRainfedArea_NDVI'][0] = 0.5
    #    adjustment_params['NorthernRiverValley_precip'][0] = 0.5
    #    adjustment_params['SouhternRainfedArea_precip'][0] = 0.5
        pass
    
    #millet
    if commodity.lower() == 'millet':
    #    #------ increase environmental time series by 20% --------
        
        e_list  = ['Kaffrine_ndvi','Kaolack_ndvi', 'Fatick_ndvi', 'Kaffrine_precip', 'Kaolack_precip',  'Fatick_precip']
        all_vars = adjustment_params.keys()
        for enviro_var in all_vars:
            adjustment_params[enviro_var][0] = 0.5
            
            
        
    #---------------------------------------------------------
    
    for var_name in new_data_vals.columns:
        z_shift , steps_back = adjustment_params[var_name]
        index = new_data_vals.columns.get_loc(var_name)
        print((z_shift * np.nanstd(new_data_vals.iloc[:, index]) ))
        new_data_vals.iloc[-steps_back:, index] = new_data_vals.iloc[-steps_back:, index] + (z_shift * np.nanstd(new_data_vals.iloc[:, index]) )
    
    
    new_data_filled = new_data_vals.fillna(mssng)
    new_data = pp.DataFrame(new_data_filled.values, var_names = new_data_filled.columns, missing_flag = mssng)
    
    predicted = pred.predict(target) #, new_data = )
    test = pred.get_test_array()[0]
    train = pred.get_train_array(target)[0]
    
    adjusted = pred.predict(target, new_data = new_data)[-predicted.size:]
    train_adjusted = pred.get_train_array(target)[0]
    test_adjusted = pred.get_test_array()[0][-predicted.size:]
    
    
    plt.scatter(test, predicted)
    plt.title(r"NRMSE = %.2f" % (np.abs(test - predicted).mean()/test.std()))
    plt.plot(test, test, 'k-')
    plt.xlabel('True test data')
    plt.ylabel('Predicted test data')
    plt.show()
    
    #plt.scatter(true_data2, adjusted)
    #plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean()/true_data.std()))
    #plt.plot(true_data, true_data, 'k-')
    #plt.xlabel('True test data')
    #plt.ylabel('adjusted test data')
    #plt.show()
    
    #---- plot time series- ---
    index = data_filled.index
    #train = pred.get_train_array(0)[0]
    #test = pred.get_test_array()[0]
    true_vals =  np.concatenate([train,test], axis = 0)
    train_range = list(range(0, train.size))
    predicted_range = list(range(train.size , train.size + test.size))
    
    predicted_range_adj = list(range(train_adjusted.size , train_adjusted.size + test_adjusted.size))
    
    # ------- set up axis -----
    f, ax1 = plt.subplots(1,1,figsize = (13,4))
    f.suptitle('Price Prediction - ' + study_market, fontsize = 14)
#    ax1.set_title('Inputs: '
    ax1.set_xlabel('Month Number', fontsize= 14)
    ax1.set_ylabel('Price z-score', fontsize = 14)
    # ------ plotting -------          
    #ax1.plot(train_range, denorm(train), color = '#75bdff', lw = 2, label = 'train values',alpha = 0.7)
    #
    ax1.plot(predicted_range, denorm(test), color = '#515ee8',lw = 2,  label = 'test values', alpha = 0.9, marker = '.')
    #ax1.plot(predicted_range_adj, denorm (test_adjusted), color = '#FF8C00',lw = 2,  label = 'adjusted test values', alpha = 0.9, marker = '.')
    ax1.plot(predicted_range, denorm(predicted), color = 'red',lw = 2, linestyle = '--', label = 'predicted (defualt)' , marker = '.')
    ax1.plot(predicted_range_adj, denorm(adjusted), color = '#327d61', lw = 2,  label = 'predicted (test scenario)', alpha = 0.9)
    ax1.axvspan( predicted_range[-1] - 12 , predicted_range[-1], alpha=0.2, color='red')
    ax1.legend()
    

    
    # ---- key parameters ----
study_market = 'Dakar' # market to predict the time series of
country = 'Senegal' # country of study, optimized for senegal 
                    # but can choose any country/commodity pair at FPMA portal: https://fpma.apps.fao.org/giews/food-prices/tool/public/#/dataset/domestic
commodity = 'Millet' #commodity to study: optimized for 'Rice' or 'Millet' in Senegal 
add_enviro = True # whether or not to add environmental variables to study. (currently only available for Senegal)
use_gee =  True # use Google Earth Engine to obtain most up to date environmental time series.
                # If True, requires valid Earth Engine login. 
                # If False, data ends in April 2021.

# ---- additional test parameters -----
min_lag, max_lag  = 1,4  # minimum and maximum lag of causal links
condition_on_my = True # whether to condition on month and year instead of directly correcting for these relations
study_variables = [] # optional list of markets to include as manual predictors to PC test
restrict_positive = True #whether or not to restrict linear regression weights to only positive values (recommend True). 
                          # (If so, environmental variables are multiplied by -1.)
interpolate = True # Interpolate time series (recommended for prediction)
max_gap= 3 # maximum gap to interpolate
minimum_size = 160 # minimum size of each price time series

# ---- display parameters
print_info = False # Whether or not to provide printed outputs for all steps of test
#print_graphs = True # print spatial and link graph

run_pred_test(country, commodity, study_market, min_lag,  max_lag, add_enviro, 
              study_variables = study_variables, m_y_conditioning = condition_on_my, 
             interpolate = False, max_gap = 3, minimum_size = minimum_size, 
              print_info = print_info,  use_gee = use_gee)
    

