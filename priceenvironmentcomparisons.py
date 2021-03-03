#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:01:53 2021

@author: Mitchell
"""

import numpy as np    
import matplotlib.pyplot as plt 
import datetime as dt
import math
import pandas as pd


fews_prices = pd.read_csv('pricedata/FEWS_price_data.csv')


#---CPI---
cpi_df = pd.read_csv('pricedata/US_CPI_1960to2019.csv', delimiter = ';')
cpi = cpi_df.CPI
#cpi.index = pd.to_datetime(cpi_df.year , format = '%Y')
cpi.index = cpi_df.year

#-----currency conversion------
curr_path = 'pricedata/FAOstatXOFtoUSD.csv'
curr_df = pd.read_csv(curr_path)
curr_df.index = curr_df.Year
# XOF per USD
curr_series = curr_df.Value

#-----global rice prices in $/mt--------
rice_global_path = '/Users/Mitchell/SenegalAnalyses/codes/twist-global-model/input/WB_grain_price/monthlyNominalGrainPrivesWB_Rice5procent_1960to2019.csv'
global_rice_prices_df = pd.read_csv(rice_global_path, delimiter='\t' )
global_rice_prices = global_rice_prices_df['RiceThai5%($/mt)']
global_rice_prices.index = pd.to_datetime(global_rice_prices.index, format = '%YM%m')
#adjusted for inflation
cpi_vals = cpi[global_rice_prices.index.year]
global_rice_adjusted = (global_rice_prices / cpi_vals.values ) * 100



#------prices of several senegalese markets in $/mt-----
def GEIWS_prices():
    rice_senegal_markets_path = 'pricedata/SenegalRice10MarketsPrices2007to2020.csv'
    rice_senegal_markets = pd.read_csv(rice_senegal_markets_path )
    rice_senegal_markets.index = pd.to_datetime(rice_senegal_markets['Date-Monthly'], format = '%y-%b')
    rice_senegal_markets.drop( 'Date-Monthly', axis = 1, inplace = True)
    rice_senegal_markets = rice_senegal_markets.astype(float)
    #adust for inflation
    cpi_vals = cpi[rice_senegal_markets.index.year]
    rice_senegal_adjusted = (rice_senegal_markets.div(cpi_vals.values , axis = 'index')) * 100
    #take average of all market prices
    rice_senegal_adjusted['Average Price (adjusted)'] = rice_senegal_adjusted.mean(axis = 1, skipna = True)
    return rice_senegal_adjusted

#------- WFP imported rice from 64 markets -----
def wfp_prices():
    wfp_imported_rice_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFP_Senegal_importedrice.csv'
    import_rice_df = pd.read_csv(wfp_imported_rice_path)
    market_names = import_rice_df.Market.drop_duplicates()
    import_rice_df.index = import_rice_df.Market
    
    start, end = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
    index = pd.date_range(start=start, end=end, freq='MS')
    markets_dict = {}
    for market in market_names:
    #    process each individual market
        market_data = import_rice_df.loc[market].dropna()
        market_years = market_data.Year
        market_data.index = pd.to_datetime(market_data.loc[:,'Year'].astype(str) + '-' + 
                                     market_data.loc[:,'Month'].astype(str), format = '%Y-%m')
        market_series = market_data.Price
    #    convert to usd / mt
        market_usd_mt = (1000 * market_series).div(curr_series[market_years].values)
        
        market_usd_mt_filled = market_usd_mt.reindex(index)
    #    add to dictionary
        markets_dict[market] = market_usd_mt_filled
        
    prices_df = pd.DataFrame.from_dict(markets_dict)
    return prices_df
    
    
    
    



#
#plt.plot(global_rice_adjusted)
#plt.suptitle('Global Rice Prices, inflation adjusted')
#plt.show()
#
#plt.plot(rice_senegal_adjusted['Average Price (adjusted)'])
#plt.suptitle('Senegal Rice Prices, inflation adjusted')
#plt.show()
#
#inter_index = global_rice_adjusted.index.intersection(rice_senegal_adjusted['Average Price (adjusted)'].index )
#senegal_minus_world = rice_senegal_adjusted['Average Price (adjusted)'][inter_index] - global_rice_adjusted[inter_index]
#plt.plot(senegal_minus_world)
#plt.axhline(0, color = 'black')
#plt.suptitle('Senegal - world, inflation adjusted')
#plt.show()







#print(global_rice_prices)