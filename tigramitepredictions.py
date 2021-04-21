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
import sklearn.neighbors
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramitecustom.models import LinearMediation, Prediction

from tigramitecausations import get_rice_dict, get_rice_df, subtract_rolling_mean, take_first_diff, adjust_seasonality, get_enviro_df, interpolate_df
from import_files import GEIWS_prices
from itertools import chain

commodity = 'rice'
study_market = 'Dakar'
add_enviro  = True
use_study_vars = True
#whether or not to restrict linear regression weights to only positive values. 
#If so, environmental variables are multiplied by -1.
restrict_positive = True

condition_on_my = False

interpolate = True
inter_max_gap = 3


s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
#------------import rice------------
mkts_dict, fao_mkts_dict = get_rice_dict()

minimum_size = 160
rice_dataframe = get_rice_df(fao_mkts_dict, None, 160, s, e)

# -----------import millet---------
senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
millet_dataframe = GEIWS_prices(senegal_millet_file)
#------set up dataframe
data = None
if commodity.lower() == 'rice' :
    data = rice_dataframe 
elif commodity.lower() == 'millet' :
    data = millet_dataframe
    
# define target index
target = list(data.columns).index(study_market)

#get stats for time series for later correction
mean = np.nanmean(data.iloc[:,target])
std = np.nanstd(data.iloc[:,target])
#define function to "denormalize" data back to real values
def denorm(z):
    return (z*std) + mean

# select specific markets to study 
study_vars = [study_market, 'Bangkok','Mumbai','SãoPaulo','NorthernRiverValley_NDVI', 'SouhternRainfedArea_NDVI',
       'NorthernRiverValley_precip', 'SouhternRainfedArea_precip']
#study_vars = millet_dataframe.columns
input_str = 'Millet - All markets and NDVI/Precip in Kaolack, Kaffrine, and Fatick' if commodity == 'millet' else 'Rice - Bangkok, Sao Paolo, Mumbai, and NDVI/Precip in SRV and Casamance'

mssng = 99999
data_clipped = data[s:e]

data_stationary = subtract_rolling_mean( adjust_seasonality( data.copy()))[s:e]

#--- environmental data
if add_enviro:
    enviro_df = get_enviro_df(commodity)[s:e]
    if restrict_positive:
        enviro_df =  - enviro_df
    data_clipped  = pd.concat([data_clipped , enviro_df], axis = 1)
    data_stationary = pd.concat([data_stationary , enviro_df], axis = 1)
    
data_interp = data_clipped.interpolate(method='linear', limit=inter_max_gap) if interpolate == True else data_clipped
data_filled = data_interp.fillna(mssng) 


def m_y_conditioning(data):
    min_lag = 1
    my_data = data.copy()[s:e]
#        if adding environmental variables
    if add_enviro:
        enviro_df = get_enviro_df(commodity, monthly_dev = False)[s:e]
        my_data = pd.concat([my_data, enviro_df], axis = 1)
        enviro_indices = [my_data.columns.get_loc(x) for x in enviro_df.columns ]
    my_data['Month'] = my_data.index.month
    my_data['Year'] = my_data.index.year
    m_y_indices = [my_data.columns.get_loc('Month'), my_data.columns.get_loc('Year')]
    my_data = my_data.interpolate(method='linear', limit=inter_max_gap) if interpolate == True else  my_data
    filled_data = my_data.fillna(mssng)
    return filled_data

if condition_on_my:
    data_filled = m_y_conditioning(data)


data_stationary_filled = data_stationary.fillna(mssng)
#data_filled = data_filled.dropna()
    
data_filled = data_filled[study_vars] if use_study_vars == True else data_filled 
data_stationary_filled = data_stationary_filled[study_vars] if use_study_vars == True else data_stationary_filled 
T, N = data_filled.shape 
dataframe = pp.DataFrame(data_filled.values, var_names = data_filled.columns, missing_flag = mssng)
d = dataframe
#dataframe = pp.DataFrame(data_filled.values, var_names = data.columns)
dataframe_stationary = pp.DataFrame(data_stationary_filled.values, var_names = data_stationary_filled.columns, missing_flag = mssng)

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
    verbosity=1
    )

pred_stationary = Prediction(dataframe= dataframe_stationary,
        cond_ind_test=ParCorr(),   #CMIknn ParCorr
        prediction_model = sklearn.linear_model.LinearRegression(positive = restrict_positive),
    data_transform=sklearn.preprocessing.StandardScaler(),
    train_indices= train_indices,
    test_indices= test_indices,
    verbosity=1
    )
# Now, we estimate causal predictors using get_predictors for the target variable 2 taking into account a maximum 
# past lag of tau_max. We use pc_alpha=None which optimizes the parameter based on the Akaike score. Note that 
# the predictors are different for each prediction horizon. For example, at a prediction horizon of steps_ahead=1 
# we get the causal parents from the model plus some others:



tau_max = 4
steps_ahead= 1 #(tau_min)
#studied_link_vars = 
selected_links = {i : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] 
                                for j in range(pred.N) if j != target ])) for i in range(pred.N)}
#selected_links = {target : list(chain.from_iterable([ [(j,k) for k in range(-tau_max, -steps_ahead + 1)] for j in range(pred.N) ])) }
    
    
predictors = pred_stationary.get_predictors(
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



new_data_vals = data_filled[study_vars].copy() if use_study_vars == True else data_filled.copy()
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
    adjustment_params['NorthernRiverValley_NDVI'][0] = 0.5
    adjustment_params['SouhternRainfedArea_NDVI'][0] = 0.5
    adjustment_params['NorthernRiverValley_precip'][0] = 0.5
    adjustment_params['SouhternRainfedArea_precip'][0] = 0.5

#millet
if commodity.lower() == 'millet':
#    #------ increase environmental time series by 20% --------
    
    e_list  = ['Kaffrine_NDVI','Kaolack_NDVI', 'Fatick_NDVI', 'Kaffrine_precip', 'Kaolack_precip',  'Fatick_precip']
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
train = pred.get_train_array(0)[0]

adjusted = pred.predict(target, new_data = new_data)[-predicted.size:]
train_adjusted = pred.get_train_array(0)[0]
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
index = data_clipped.index
#train = pred.get_train_array(0)[0]
#test = pred.get_test_array()[0]
true_vals =  np.concatenate([train,test], axis = 0)
train_range = list(range(0, train.size))
predicted_range = list(range(train.size , train.size + test.size))

predicted_range_adj = list(range(train_adjusted.size , train_adjusted.size + test_adjusted.size))

# ------- set up axis -----
f, ax1 = plt.subplots(1,1,figsize = (13,4))
f.suptitle('Price Prediction - Dakar', fontsize = 14)
ax1.set_title('Inputs: ' + input_str, fontsize = 12)
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



