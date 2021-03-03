#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:40:57 2021

@author: Mitchell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from import_files import wfp_prices, GEIWS_prices, exchange_rates, get_ndvi_ts, get_precip_ts, get_flood_ts, get_enviro_ts
import statsmodels.tsa.vector_ar.vecm.VECM as VECM

#----- prices from WFP data, senegal markets
wfp_imported_rice_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFP_Senegal_importedrice.csv'
senegal_wfp_dataframe, senegal_mkts_dict = wfp_prices(wfp_imported_rice_path, minimum_size = 0)
mkts_dict = senegal_mkts_dict.copy()

#WFP farafenni, basse, gambia:

gambia_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFPFarafenniBasseLongGrain.csv'
gambia_df, gambia_dict = wfp_prices(gambia_path, minimum_size = 0, combine = True)
del gambia_dict['CombinedAverage']
for key in gambia_dict.keys():
    mkts_dict[key] = gambia_dict[key]

#---- WFP, banjul, Gambia
banjul_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFP_2021Feb04_Gambia_FoodPricesData_LONGGRAIN.csv'
banjul_df, banjul_dict = wfp_prices(banjul_path, minimum_size = 0, combine = True)
mkts_dict['Banjul'] = banjul_dict['CombinedAverage']

bissau_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFP_2021Feb04_Guinea-Bissau_FoodPricesData.csv'
bissau_df, bissau_dict = wfp_prices(bissau_path, minimum_size = 0, combine = True)
mkts_dict['Bisseau'] = bissau_dict['CombinedAverage']


conakry_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/GuineaConakryAllMarkets.csv'
conakry_df, conakry_dict = wfp_prices(conakry_path, minimum_size = 0, combine = True)
mkts_dict['Conakry'] = bissau_dict['CombinedAverage']

fao_senegal_prices, fao_border_prices, fao_international_prices = GEIWS_prices()
for df in fao_senegal_prices, fao_border_prices,fao_international_prices:
    for column in df.columns:
        series = df[column]
        mkts_dict[column] = series
        
        
ndvi_file = 'envirodata/NDVIcities2.csv'
ndvi_dict =  get_enviro_ts(ndvi_file, 'NDVI',split_date = 'True')


s , e =  pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
#minimum time series size
size_th = 160
study_markets =  ['Dakar', 'Saint-Louis', 'Dagana','Nouakchott','Kayes','Tambacounda','Touba','Bakel',
                     'Banjul','Farafenni', 'Zigiunchor','Kolda', 'Basse Santa su', 'Diaobe', 'Bisseau','Conakry', 'Kaolack', 'Bangkok','Mumbai','SãoPaulo']
final_dict  = {x : mkts_dict[x] for x in list(mkts_dict.keys()) if len(mkts_dict[x][s:e].dropna()) >= size_th}

print(final_dict.keys())


formula = 'ts1 ~ Literacy + Wealth + Region'























