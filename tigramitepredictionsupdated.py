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
from datetime import date
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
restrict_positive = False
#
#condition_on_my = True
#
#interpolate = True
#inter_max_gap = 3
#

#
#

###------------import rice------------
##mkts_dict, fao_mkts_dict = get_rice_dict()
##
#minimum_size = 160
#
#max_lag = 4
#min_lag = 1 #(tau_min)

def denorm(z):

    return (z*std) + mean

#rice_dataframe = get_rice_df(fao_mkts_dict, None, 160, s, e)
#
## -----------import millet---------
##senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
##millet_dataframe = GEIWS_prices(senegal_millet_file)
#
#millet_prices = pd.DataFrame.from_dict(extract_giews(country = 'Senegal', commodity = 'Millet', min_size = minimum_size))

#allows for saving enviro data over multiple runs
if 'enviro_data_dict' not in dir():
    enviro_data_dict = {} 

def run_pred_test(country, commodity, study_market, steps_ahead,  tau_max, add_enviro, start,end,
            shock_period = None, shock_values = None, study_variables = None,  m_y_conditioning = True, 
             interpolate = True, max_gap = 3, print_info = False,  use_gee = True, minimum_size = 160):
    ''' 
    Code to run predictive test based on tigramite PCMCI network framework. 
    Optimized for use in Senegal with Rice and Millet but applicable to any country/commodity combination
    '''
    
    global data
    data = create_data(country, commodity,min_size = minimum_size)
    #  set up environmental dataframe if valid   
    if add_enviro: # make dataframe for NDVI and Precip over regions of commodity growth, 
                   # flip to negative if restricting lin regression to positive       
        if (country, commodity) in enviro_data_dict.keys():
            enviro_df = enviro_data_dict[(country, commodity)]
        else:
            if use_gee:
                enviro_df =   make_enviro_data(country, commodity) 
            else:
                enviro_df = pd.read_csv('envirodata/{}-fullenviro.csv'.format(commodity.lower()) )
                enviro_df.index = pd.to_datetime(enviro_df.iloc[:,0], format = '%Y-%m-%d')
                enviro_df =  enviro_df.drop(columns = enviro_df.columns[0])
            enviro_data_dict[(country, commodity)] = enviro_df
        enviro_df = - enviro_df if restrict_positive == True else enviro_df
        
        
    # define target index
    target = list(data.columns).index(study_market)
    
    #get stats for time series for later correction
#    global mean
#    global std
#    mean = np.nanmean(data.iloc[:,target])
#    std = np.nanstd(data.iloc[:,target])
#    def denorm(z):
#        return (z*std) + mean
    
    #define function to "denormalize" data back to real values
    


    #study_vars = millet_dataframe.columns
#    input_str = 'Millet - All markets and NDVI/Precip in Kaolack, Kaffrine, and Fatick' if commodity == 'millet' else 'Rice - Bangkok, Sao Paolo, Mumbai, and NDVI/Precip in SRV and Casamance'
#    
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

        t1 = m_y_data.copy()
            
        m_y_data['Month'] = m_y_data.index.month
        m_y_data['Year'] = m_y_data.index.year
        m_y_indices = [m_y_data.columns.get_loc('Month'), m_y_data.columns.get_loc('Year')]
        m_y_data = m_y_data.interpolate(method='linear', limit=max_gap) if interpolate == True else  m_y_data
        global data_sliced
#        print(m_y_data[study_market].first_valid_index() , m_y_data[study_market].first_valid_index())
        data_sliced = m_y_data[m_y_data[study_market].first_valid_index()  : m_y_data[study_market].last_valid_index()]
    
        
        
        
    else:
        pass
    
    data_filled = data_sliced.fillna(mssng)
      
#        get proper date array which considers missing values
#    bool_date_arr = np.ones((data_sliced.shape[0]), dtype=bool)
#    bool_date_arr[:tau_max ] = False
#    for i, row in enumerate(data_sliced.values):
#        if np.isnan(row).any():
#            prev_index = i - tau_max - 1 if i > tau_max + 1 else 0
##            mark that date and dates all before within tau max as false
#            bool_date_arr[prev_index : i + 1] = False
#    global new_date_arr  
#    new_date_arr = data_sliced.index[bool_date_arr]
#    
#    global dakar_new_dates
#    dakar_new_dates = data_sliced.loc[new_date_arr].Dakar
#            
            
            
            
        

            
        
    data_filled = data_filled[study_variables] if study_variables == True else data_filled   

    
    
    T, N = data_filled.shape
    
    selected_links = {i : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] 
                                    for j in range(N) if j != target ])) for i in range(N)}
    
    dataframe = pp.DataFrame(data_filled.values, var_names = data_filled.columns, missing_flag = mssng)
#    data_enviro_adjusted = 
#    dataframe_enviro_adj = pp.DataFrame(data_filled.values, var_names = data_filled.columns, missing_flag = mssng)
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
    
    global mean
    global std
    mean = np.nanmean(data_sliced.iloc[:,target])
    std = np.nanstd(data_sliced.iloc[:,target])
    def denorm(z):
        return (z*std) + mean
    
    global train_mean
    global train_std
    
    train_mean  = np.nanmean(data_sliced.iloc[:int(0.8*T),target])
    train_std = np.nanstd(data_sliced.iloc[:int(0.8*T),target])
    def train_denorm(z):
        return (z*train_std) + train_mean
    
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
    
    
    
    
    #    get date array to go with prediction arrays
    bool_date_arr = np.ones((data_sliced.shape[0]), dtype=bool)
    bool_date_arr[:tau_max ] = False
    
    for i, row in enumerate(data_sliced.values):
        if np.isnan(row).any():
            future_index = i + tau_max + 1
            bool_date_arr[i : future_index] = False
            
#    updated date array for prediction
    new_date_arr = data_sliced.index[bool_date_arr]
    
    
    
    
#    APPLY SHOCKS
    
    new_data_vals = data_filled[study_variables].copy() if study_variables == True else data_filled.copy()
    new_data_vals = new_data_vals.replace(mssng,np.nan)
    
    #-----adjustment for each variable --------
    # dict of form:   variable : [ z_shift (float),  steps back (int, how far back to apply shock) ]
    # z_shift is how many z-scores to shift data
    if shock_values:
        for var_item in shock_values.items():
            var_name, (z_shift , steps_back) = var_item
            try:
                indices = [new_data_vals.columns.get_loc(var_name)]
            except KeyError:
                indices = [new_data_vals.columns.get_loc(x) for x in new_data_vals.columns if var_name.lower() in x.lower()]
                
            for index in indices:
                new_data_vals.iloc[-steps_back:, index] = (new_data_vals.iloc[-steps_back:, index] + 
                                  (z_shift * np.nanstd(new_data_vals.iloc[:, index]) ))
            
    
    
    
    
#    make new data array with shocks 
    new_data_filled = new_data_vals.fillna(mssng)
    
    new_data = pp.DataFrame(new_data_filled.values, var_names = new_data_filled.columns, missing_flag = mssng)
    predicted = pred.predict(target) #, new_data = )
    test = pred.get_test_array()[0]
    train = pred.get_train_array(target)[0]
    
    adjusted = pred.predict(target, new_data = new_data)[-predicted.size:]
    train_adjusted = pred.get_train_array(target)[0]
    test_adjusted = pred.get_test_array()[0][-predicted.size:]
    
    nrmse = (np.abs(test - predicted).mean()/test.std())
    plt.scatter(test, predicted)
    plt.title(r"NRMSE = %.2f" % (np.abs(test - predicted).mean()/test.std()))
    plt.plot(test, test, 'k-')
    plt.xlabel('True test data')
    plt.ylabel('Predicted test data')
    plt.show()
    
    
    #---- plot time series- ---
    index = data_filled.index
    #train = pred.get_train_array(0)[0]
    #test = pred.get_test_array()[0]

    true_vals =  np.concatenate([train,test], axis = 0)
    train_range = list(range(0, train.size))
    predicted_range = list(range(train.size , train.size + test.size))
#    print(train_range, predicted_range)
#    predicted_range_adj = list(range(train_adjusted.size , train_adjusted.size + test_adjusted.size))
    
    # ------- set up axis -----
    f, ax1 = plt.subplots(1,1,figsize = (13,4))
    f.suptitle('{} Price Prediction - {}'.format(commodity, study_market) , fontsize = 14)
#    ax1.set_title('Inputs: '
    ax1.set_xlabel('Date', fontsize= 14)
    ax1.set_ylabel('Price ($/mt)', fontsize = 14)
    # ------ plotting -------          
    
    #
    ax1.plot(new_date_arr[predicted_range], denorm(test), color = '#515ee8',lw = 2,  label = 'test values', alpha = 0.9, marker = '.')
    #ax1.plot(predicted_range_adj, denorm (test_adjusted), color = '#FF8C00',lw = 2,  label = 'adjusted test values', alpha = 0.9, marker = '.')
    ax1.plot(new_date_arr[predicted_range], denorm(predicted), color = 'red',lw = 2, linestyle = '--', label = 'predicted (defualt)' , marker = '.')
#    ax1.plot(new_date_arr[predicted_range], denorm(adjusted), color = '#327d61', lw = 2,  label = 'predicted (test scenario)', alpha = 0.9)
#    ax1.axvspan( new_date_arr[-] , new_date_arr[-1], alpha=0.2, color='red')
    ax1.legend()
    print("DIFFERENCE ", np.max(denorm(adjusted)) - np.max(denorm(predicted)))
    temp_fig = plt.gcf()
    
    plt.show()
    ax1 = temp_fig.axes[0]
    ax1.plot(new_date_arr[train_range], denorm(train), color = '#75bdff', lw = 2, label = 'train values',alpha = 0.7)
    plt.show()
    
    return nrmse
    

    
#     # ---- key parameters ----
# study_market = 'Dakar' # market to predict the time series of
# country = 'Senegal' # country of study, optimized for senegal 
#                     # but can choose any country/commodity pair at FPMA portal: https://fpma.apps.fao.org/giews/food-prices/tool/public/#/dataset/domestic
# commodity = 'Rice' #commodity to study: optimized for 'Rice' or 'Millet' in Senegal 
# add_enviro = True # whether or not to add environmental variables to study. (currently only available for Senegal)
# use_gee =  True # use Google Earth Engine to obtain most up to date environmental time series.
#                 # If True, requires valid Earth Engine login. 
#                 # If False, data ends in April 2021.
                
# #shock_values = {'Bangkok': (1,12), 'Mumbai':(1,12),
# #                'Sao Paolo': (1,12)}   


# common_shift = (2,12)
# shock_values = {'precip': common_shift, 'ndvi':common_shift, 'spei':common_shift, 'pdsi':common_shift}
                

# # ---- additional test parameters -----
# s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(date.today().year, date.today().month, 1) # start and end of calculation
# min_lag, max_lag  = 1,4  # minimum and maximum lag of causal links
# condition_on_my = True # whether to condition on month and year instead of directly correcting for these relations
# study_variables = [] # optional list of markets to include as manual predictors to PC test
# restrict_positive = True #whether or not to restrict linear regression weights to only positive values (recommend True). 
#                           # (If so, environmental variables are multiplied by -1.)
# interpolate = True # Interpolate time series (recommended for prediction)
# max_gap= 3 # maximum gap to interpolate
# minimum_size = 160 # minimum size of each price time series

# # ---- display parameters
# print_info = False # Whether or not to provide printed outputs for all steps of test
# #print_graphs = True # print spatial and link graph


# #run_pred_test(country, commodity, study_market, min_lag,  max_lag, add_enviro, s,e,
# #             
# #              study_variables = study_variables, m_y_conditioning = condition_on_my, 
# #             interpolate = True, max_gap = 3, minimum_size = minimum_size, 
# #              print_info = print_info,  use_gee = use_gee)

# # ------ Getting Average NRMSE ---------
# rice_markets = ['Dakar', 'Diourbel', 'Kaolack', 'SaintLouis', 'Tambacounda', 'Thies',  'Nouakchott']
# millet_markets = ['Dakar', 'Diourbel', 'Fatick', 'Kaolack', 'Kolda', 'Louga', 'Matam', 'SaintLouis', 'Tambacounda', 'Thies', 'Ziguinchor']
# millet_accuracies = []
# rice_accuracies = []
# for mkt in millet_markets:
#     nrmse = run_pred_test(country, 'Millet', mkt, min_lag,  max_lag, add_enviro, s,e,
#               study_variables = study_variables, m_y_conditioning = condition_on_my, 
#              interpolate = True, max_gap = 3, minimum_size = minimum_size, 
#               print_info = print_info,  use_gee = use_gee)
#     millet_accuracies.append(nrmse)
    
# for mkt in rice_markets:
#     nrmse = run_pred_test(country, 'Rice', mkt, min_lag,  max_lag, add_enviro, s,e,
#               study_variables = study_variables, m_y_conditioning = condition_on_my, 
#              interpolate = True, max_gap = 3, minimum_size = minimum_size, 
#               print_info = print_info,  use_gee = use_gee)
#     rice_accuracies.append(nrmse)
    
# f, ( ax1, ax2) = plt.subplots(1,2,figsize = (10,5))
# ax1.hist(rice_accuracies, bins = 10)
# ax1.hist(millet_accuracies, bins = 10)
# # -------------------


    


#ax1.plot(new_date_arr[train_indices], denorm(train), color = '#75bdff', lw = 2, label = 'train values',alpha = 0.7)
##    ax1.plot(dakar_new_dates.values, color = '#515ee8', lw = 2, label = 'dak test',alpha = 0.7)
#
##
#ax1.plot(new_date_arr[test_indices], denorm(test), color = '#515ee8',lw = 2,  label = 'test values', alpha = 0.9, marker = '.')
##    ax1.plot(new_date_arr[test_indices], denorm(test_adjusted), color = '#FF8C00',lw = 2,  label = 'adjusted test values', alpha = 0.9, marker = '.')
#ax1.plot(new_date_arr[test_indices], denorm(predicted), color = 'red',lw = 2, linestyle = '--', label = 'predicted (defualt)' , marker = '.')
#ax1.plot(new_date_arr[test_indices], denorm(adjusted), color = '#327d61', lw = 2,  label = 'predicted (test scenario)', alpha = 0.9)
    





            
#    steps_back = 12
#    adjustment_params = { var_name : [0, steps_back] for var_name in new_data_vals.columns}     
    
    
    #data is adjusted here:
    
    #------ increase environmental time series by half a z-score --------
    #rice
#    if commodity.lower() == 'rice':
#    #    adjustment_params['NorthernRiverValley_NDVI'][0] = 0.5
#    #    adjustment_params['SouhternRainfedArea_NDVI'][0] = 0.5
#    #    adjustment_params['NorthernRiverValley_precip'][0] = 0.5
#    #    adjustment_params['SouhternRainfedArea_precip'][0] = 0.5
#        pass
    
    #millet
#    if commodity.lower() == 'millet':
#    #    #------ increase environmental time series by 20% --------
#        
#        e_list  = ['Kaffrine_ndvi','Kaolack_ndvi', 'Fatick_ndvi', 'Kaffrine_precip', 'Kaolack_precip',  'Fatick_precip']
#        all_vars = adjustment_params.keys()
#        for enviro_var in all_vars:
#            adjustment_params[enviro_var][0] = 0.5
            
            
        
    #---------------------------------------------------------
    
#    for var_name in new_data_vals.columns:
#        z_shift , steps_back = adjustment_params[var_name]
#        index = new_data_vals.columns.get_loc(var_name)
#        print((z_shift * np.nanstd(new_data_vals.iloc[:, index]) ))
#        new_data_vals.iloc[-steps_back:, index] = new_data_vals.iloc[-steps_back:, index] + (z_shift * np.nanstd(new_data_vals.iloc[:, index]) )    