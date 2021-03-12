#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:07:34 2021

@author: mitchell
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline     
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
import sklearn
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

from tigramitecausations import get_rice_dict, get_rice_df, subtract_rolling_mean, take_first_diff, adjust_seasonality
from import_files import GEIWS_prices

mkts_dict, fao_mkts_dict = get_rice_dict()
s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
minimum_size = 160
rice_dataframe = get_rice_df(fao_mkts_dict, None, 160, s, e)

# -----------import millet---------
senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
millet_dataframe = GEIWS_prices(senegal_millet_file)







np.random.seed(42)
T = 150
links_coeffs = {0: [((0, -1), 0.6)],
                1: [((1, -1), 0.6), ((0, -1), 0.8)],
                2: [((2, -1), 0.5), ((1, -1), 0.7)],  # ((0, -1), c)],
                }
N = len(links_coeffs)
data, true_parents = pp.var_process(links_coeffs, T=T)
dataframe = pp.DataFrame(data, var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$'])


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


# Now, we estimate causal predictors using get_predictors for the target variable 2 taking into account a maximum 
# past lag of tau_max. We use pc_alpha=None which optimizes the parameter based on the Akaike score. Note that 
# the predictors are different for each prediction horizon. For example, at a prediction horizon of steps_ahead=1 
# we get the causal parents from the model plus some others:

target = 2
tau_max = 5
predictors = pred.get_predictors(
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




