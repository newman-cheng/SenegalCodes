#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:17:05 2021

@author: Mitchell
"""

import requests
import numpy as np
import pandas as pd
from datetime import date

#import json

#json with all data
all_jsons = requests.get('https://fpma.apps.fao.org/giews/food-prices/tool/api/v1/series').json()

cntry = 'Senegal'
#download_commodity = 'rice (imported)'
cmdty = 'millet'



def extract_giews(country = None, commodity = None, market = None, min_size = 0):
    ''' gets data from FPMA GIEWS website directly based on country, commodity, and market
    
    Parameters
    ----------------------
    country (str) - optional
        country to filter by
    commodity (str) - optional
        commodity to filter by
    market (str) - optional
        market to filter by
    '''
    
#    if str(country).lower() == 'senegal' and str(commodity).lower() == 'rice':
#        commodity = 'Rice (imported)'
    
    #data which fulfills the requirements passed through the function
    selected_data = [data_dict for data_dict in all_jsons if 
                     ( (country == None or data_dict['countryName'].lower() == country.lower()) and  
                       (commodity == None or commodity.lower() in  data_dict['commodity'].lower() ) and 
                       (market == None or data_dict['market'].lower() == market.lower()))]
        #    extract data time series from selected_data
    
    data_dict = {}
    for mkt_dict in selected_data:
        mkt_name = mkt_dict['market']
#        print(mkt_name)
        if mkt_name =='Zguinchor':
#            print('test')
            mkt_name = 'Ziguinchor' #fix mispelling
        link = [curr_dict['href'] for curr_dict in mkt_dict['links'] if curr_dict['rel'] == 'monthly_usd_tonne'][0]
        mkt_json = requests.get(link).json()
        data_array = np.array(mkt_json['data'])
        global series
        series = pd.Series(data_array[:,1], name = mkt_name, dtype = float)
        series.index = pd.to_datetime( [ pd.Timestamp(date, unit = 'ms') for date in data_array[:,0] ] )
        
#        reindex series
        start, end = pd.Timestamp(2000,1,1) , pd.Timestamp(date.today().year, date.today().month, 1)
        new_index = pd.date_range(start=start, end=end, freq='MS')
        reindexed = series.reindex(new_index)
#        add to dict
        if series.dropna().shape[0] >= min_size:
            data_dict[mkt_name] = reindexed
        
    
    
    return data_dict 

def get_attribute(attribute):
    ''' gets list of available attribute FPMA GIEWS website 
    
    Parameters
    ----------------------
    attribute (str)
        attribute to get list of.
        ex: 'countryName','market','commodity', etc.
    '''
    attribute_list = np.unique([market_dict[attribute] for market_dict in all_jsons])
    return attribute_list 




#r= extract_giews(country = 'Senegal', commodity = 'rice')


