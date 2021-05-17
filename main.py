#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:46:40 2021

@author: Mitchell
"""

from tigramitecausationsupdated import run_test as causation_test

# ----- Run Causation -----

country = 'Senegal'
commodity = 'Millet'
#    
#country = 'Niger'
#commodity = 'Rice'
#
#country = 'Tanzania'
#commodity = 'Rice'

FDR_bool = False
min_lag, max_lag  = 1,4
add_enviro = True
minimum_size = 150
alpha = 0.05
m_y_conditioning = True 
interpolate = True
max_gap= 3
stationarity_method = 'firstdifference'
print_info = False

link_df = causation_test(country, commodity, FDR_bool, min_lag, max_lag, add_enviro, alpha, m_y_conditioning = m_y_conditioning, interpolate = interpolate,
        minimum_size = minimum_size, max_gap= max_gap, stationarity_method = 'firstdifference', print_info = False)



# ---- RUN PREDICTION ----