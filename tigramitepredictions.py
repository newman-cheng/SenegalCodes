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
plt.style.use('ggplot')
import sklearn
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

from tigramitecausations import get_rice_dict, get_rice_df, subtract_rolling_mean, take_first_diff, adjust_seasonality, get_enviro_df
from import_files import GEIWS_prices


commodity = 'rice'
study_market = 'Dakar'
add_enviro  = True
use_study_vars = True


s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
#------------import rice------------
mkts_dict, fao_mkts_dict = get_rice_dict()

minimum_size = 160
rice_dataframe = get_rice_df(fao_mkts_dict, None, 160, s, e)

# -----------import millet---------
senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
millet_dataframe = GEIWS_prices(senegal_millet_file)
#------set up dataframe

data = rice_dataframe if commodity.lower() == 'rice' else millet_dataframe


# select specific markets to study 
study_vars = [study_market, 'Bangkok','Mumbai','SãoPaulo','NorthernRiverValley_NDVI', 'SouhternRainfedArea_NDVI',
       'NorthernRiverValley_precip', 'SouhternRainfedArea_precip']

mssng = 99999
data_clipped = data[s:e]

data_stationary = subtract_rolling_mean( adjust_seasonality( data.copy()))[s:e]

#--- environmental data
if add_enviro:
    enviro_df = get_enviro_df()[s:e]
    data_clipped  = pd.concat([data_clipped , enviro_df], axis = 1)
    data_stationary = pd.concat([data_stationary , enviro_df], axis = 1)
    
data_filled = data_clipped.fillna(mssng)
data_stationary_filled = data_stationary.fillna(mssng)
#data_filled = data_filled.dropna()
    
data_filled = data_filled[study_vars] if use_study_vars == True else data_filled 
data_stationary_filled = data_stationary_filled[study_vars] if use_study_vars == True else data_stationary_filled 

T, N = data_filled.shape
dataframe = pp.DataFrame(data_filled.values, var_names = data.columns, missing_flag = mssng)
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

pred = Prediction(dataframe=dataframe,
        cond_ind_test=ParCorr(),   #CMIknn ParCorr
        prediction_model = sklearn.linear_model.LinearRegression(),
#         prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
        # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
    data_transform=sklearn.preprocessing.StandardScaler(),
    train_indices= range(int(0.8*T)),
    test_indices= range(int(0.8*T), T),
    verbosity=1
    )

pred_stationary = Prediction(dataframe= dataframe_stationary,
        cond_ind_test=ParCorr(),   #CMIknn ParCorr
        prediction_model = sklearn.linear_model.LinearRegression(),
#         prediction_model = sklearn.gaussian_process.GaussianProcessRegressor(),
        # prediction_model = sklearn.neighbors.KNeighborsRegressor(),
    data_transform=sklearn.preprocessing.StandardScaler(),
    train_indices= range(int(0.8*T)),
    test_indices= range(int(0.8*T), T),
    verbosity=1
    )
# Now, we estimate causal predictors using get_predictors for the target variable 2 taking into account a maximum 
# past lag of tau_max. We use pc_alpha=None which optimizes the parameter based on the Akaike score. Note that 
# the predictors are different for each prediction horizon. For example, at a prediction horizon of steps_ahead=1 
# we get the causal parents from the model plus some others:

target = list(rice_dataframe.columns).index(study_market)
target = 0
tau_max = 4
predictors = pred_stationary.get_predictors(
                  selected_targets=[target],
                  steps_ahead=1,
                  tau_max=tau_max,
                  pc_alpha=None
                  )
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

predicted = pred.predict(target)
true_data = pred.get_test_array()[0]

plt.scatter(true_data, predicted)
plt.title(r"NRMSE = %.2f" % (np.abs(true_data - predicted).mean()/true_data.std()))
plt.plot(true_data, true_data, 'k-')
plt.xlabel('True test data')
plt.ylabel('Predicted test data')
plt.show()

#---- plot time series- ---
index = data_clipped.index
train = pred.get_train_array(0)[0]
test = pred.get_test_array()[0]
true_vals =  np.concatenate([train,test], axis = 0)
train_range = list(range(0, train.size))
predicted_range = list(range(train.size , train.size + test.size))

# ------- set up axis -----
f, ax1 = plt.subplots(1,1,figsize = (13,4))
f.suptitle('Price Prediction - Dakar', fontsize = 14)
ax1.set_title('Inputs: Bangkok, Mumbai, Sao Paolo, NDVI and Precipitation in SRV and Casamance', fontsize = 12)
ax1.set_xlabel('Month Number', fontsize= 14)
ax1.set_ylabel('Price z-score', fontsize = 14)
# ------ plotting -------          
ax1.plot(train_range, train, color = '#75bdff', lw = 2, label = 'train values',alpha = 0.7)
ax1.plot(predicted_range, test, color = '#515ee8',lw = 2,  label = 'test values', alpha = 0.9)
ax1.plot(predicted_range, predicted, color = 'red',lw = 2, linestyle = '--', label = 'predicted')
ax1.legend()



