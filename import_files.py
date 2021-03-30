#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:33:23 2021

@author: Mitchell
"""

import pandas as pd
import numpy as np

#----- yearly currency conversion------
def get_currency():
    curr_path = 'pricedata/FAOstatXOFtoUSD.csv'
    curr_df = pd.read_csv(curr_path)
    curr_df.index = curr_df.Year
    # XOF per USD
    curr_series = curr_df.Value
    return curr_series

# returns XOF to USD exchange rate
def exchange_rates():
    oanda_file = 'pricedata/oanda_com-monthly-exchangerates-2002-2013.csv'

    oanda_df = pd.read_csv(oanda_file)
    oanda_df.index = pd.to_datetime(oanda_df.Date, format = '%y-%b')
    XOF1 = oanda_df['CFA Westfranc']
    XOF1 = XOF1.sort_index()
    
    wfp_file = 'pricedata/monthly-exchangerates-2007-2020.csv'
    wfp_df  =pd.read_csv(wfp_file)
    wfp_df.index = pd.to_datetime(wfp_df['Date-Monthly'], format = '%y-%b')
    XOF2 = wfp_df['CFAFranc/USD'].replace('#DIV/0!', np.nan).astype(float)
    XOF2 = XOF2.sort_index()
    XOF = XOF1.append(XOF2[pd.Timestamp(2013,11,2):]).interpolate()
    return XOF


#------prices of several senegalese markets in $/mt-----
senegal_file = 'pricedata/SenegalRice10MarketsPrices2007to2020.csv'
border_file = 'pricedata/SurroundingMarketsSenegalPrices.csv'
international_file = 'pricedata/GEIWS_international_rice.csv'
def GEIWS_prices(file):

    commod_markets = pd.read_csv(file )
#        print(rice_markets)
    try:
        commod_markets.index = pd.to_datetime(commod_markets['Date-Monthly'], format = '%y-%b')
    except ValueError:
        commod_markets.index = pd.to_datetime(commod_markets['Date-Monthly'], format = '%b-%y')
    commod_markets.drop('Date-Monthly', axis = 1, inplace = True)
    commod_markets = commod_markets.astype(float)
    commod_markets.columns = [x.replace(' ','').split(',')[2] for x in commod_markets.columns]
#        reindex
    start, end = pd.Timestamp(2000,1,1) , pd.Timestamp(2020,12,31)
#        global index
    index = pd.date_range(start=start, end=end, freq='MS')
    commod_markets = commod_markets.reindex(index)
    return commod_markets
    
    #adust for inflation
# #        cpi_vals = get_currency()[rice_markets.index.year]
# #        rice_adjusted = (rice_markets.div(cpi_vals.values , axis = 'index')) * 100
#     #take average of all market prices
    
# #        rice_markets['Average Price (adjusted)'] = pd.Series(rice_markets.mean(axis = 1, skipna = True))
# #        print(type(rice_markets['Average Price (adjusted)']))
#     results_tup[i] = rice_markets
        
#     return tuple(results_tup) #senegal_prices, border_prices

senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
GEIWS_prices(senegal_millet_file)

def wfp_prices(path, minimum_size = 50, combine = False, curr_exchange = True):
    
    import_rice_df = pd.read_csv(path)
#    print(import_rice_df)
#    global market_names
    market_names = import_rice_df.Market.drop_duplicates()
    import_rice_df.index = import_rice_df.Market
    global curr_series
    curr_series = exchange_rates()
    start, end = pd.Timestamp(2002,1,1) , pd.Timestamp(2020,12,31)
#    global index
    index = pd.date_range(start=start, end=end, freq='MS')
    markets_dict = {}
    for market in market_names:
    #    process each individual market
        market_data = import_rice_df.loc[market].dropna().drop_duplicates()
#        set some size limit for each time series
        if len(market_data) >= minimum_size:
            
            market_data.index = pd.to_datetime(market_data.loc[:,'Year'].astype(str) + '-' + 
                                         market_data.loc[:,'Month'].astype(str), format = '%Y-%m')
            market_series = market_data.Price[start:end]
        #    convert to usd / mt
#            global market_usd_mt
            
            market_usd_mt = (1000 * market_series).div(curr_series[market_series.index].values) if curr_exchange == True else market_series
#            print(market_usd_mt.index)
            market_usd_mt_filled = market_usd_mt.reindex(index)
        #    add to dictionary
            markets_dict[market] = market_usd_mt_filled
        
    prices_df = pd.DataFrame.from_dict(markets_dict)
    if combine:
        prices_df['CombinedAverage'] = prices_df.mean(axis = 1, skipna = True)
        markets_dict['CombinedAverage'] = prices_df['CombinedAverage']
    return prices_df, markets_dict

#------------ WFP Prices from Mamina-----------
def wfp_mamina_prices(month_resample = True, international = False, minimum_size = None):
    s,e = pd.Timestamp(2000,1,1) , pd.Timestamp(2020,12,31)
    rice_dict = {}
    millet_dict = {}
    mamina_price  = pd.read_csv('pricedata/MaminaData.csv')
    mamina_price.index = mamina_price.DEPARTEMENT
    for mkt in mamina_price.index.drop_duplicates():
        select_data = mamina_price.loc[mkt]
        select_data.index = pd.to_datetime(select_data.DATE, format = '%d-%b-%y')
        millet_ts = select_data['MIL_DETAIL']
        rice_ts =  select_data['RIZ_IMP_BR ORD.']
    
        if month_resample == True:
            dt_index = pd.date_range(start=s, end=e, freq = 'MS')
            millet_ts = millet_ts.resample('MS').median().reindex(dt_index)
            rice_ts = rice_ts.resample('MS').median().reindex(dt_index)
        if millet_ts.dropna().size >= minimum_size:
            millet_dict[mkt.capitalize()] = millet_ts
        if rice_ts.dropna().size >= minimum_size:
            rice_dict[mkt.capitalize()] = rice_ts
    international_file = 'pricedata/GEIWS_international_rice.csv'
    mam_rice_dataframe = pd.concat([pd.DataFrame.from_dict(rice_dict), GEIWS_prices(international_file)], axis = 1) if international == True else pd.DataFrame.from_dict(rice_dict)
    mam_millet_dataframe = pd.DataFrame.from_dict(millet_dict)
    return mam_rice_dataframe, mam_millet_dataframe


#------- Plots comparing time series ----------
#s, e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
#mam_rice_dataframe, mam_millet_dataframe = wfp_mamina_prices(minimum_size = 240)
#fao_senegal = GEIWS_prices('pricedata/SenegalRice10MarketsPrices2007to2020.csv')[s:e]
#fao_senegal = fao_senegal.multiply(exchange_rates()[fao_senegal.index].values, axis = 0) / 1000
#
#
#senegal_wfp_dataframe, senegal_mkts_dict = wfp_prices('pricedata/WFP_Senegal_importedrice.csv', minimum_size = 0, curr_exchange = False)
#
#import matplotlib.pyplot as plt
#mkt = 'SaintLouis'
#f, ax = plt.subplots(1,1,figsize = (16,4)) 
#ax.set_ylabel('XOF/kg')
#f.suptitle(mkt+' Rice Prices', fontsize = 20)
#lw = 2.5
#mam_rice_dataframe['St.louis'][s:e].plot(ax = ax, legend = False, lw = lw)
#senegal_wfp_dataframe['Saint-Louis'][s:e].plot(ax=ax, legend  = False, lw = lw)
#fao_senegal[mkt].plot(ax=ax, legend = False, linestyle = '--', lw = lw)
#f.legend(['Data from Mamina','WFP VAM data','FAO GEIWS data'])
#





#prices_df, markets_dict = wfp_prices('/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFP_Senegal_importedrice.csv')
#dakar1= prices_df['Dakar'].dropna()
#senegal, border = GEIWS_prices()



    
def get_ndvi_ts():
    file = 'envirodata/NDVIcities2.csv'
    df = pd.read_csv(file)
    df.Date = df.Date.str.slice(start = 2)
    ndvi_dict = {}
    for name in df.Name.drop_duplicates():
        filtered_df = df[df.Name == name]
        ts = filtered_df.NDVI
        ts.index = pd.to_datetime(filtered_df.Date, format = '%Y_%m_%d')
        ndvi_dict[name + '_ndvi'] = ts.dropna()
    return ndvi_dict

def get_precip_ts(monthly_deviations = False):
    file = 'envirodata/ChirpsMonthlySums.csv'
    df = pd.read_csv(file)
    precip_dict = {}
    for name in df.City.drop_duplicates():
        filtered_df = df[df.City == name]
        ts = filtered_df.montlyPrecipSumAvg
        ts.index = pd.to_datetime(filtered_df.Date, format = '%m_%d_%Y')
        
        monthly_ts = ts.resample('MS').mean() 
        if monthly_deviations == True:
            month_arr = monthly_ts.index.month
            months = [x+1 for x in range(12)]
            month_means = []
            month_std = []
        
            for month in months:
                month_mean = monthly_ts[month_arr == month].mean()
                month_std_dev = monthly_ts[month_arr == month].std()
                month_means.append(month_mean)
                month_std.append(month_std_dev)
            month_means = pd.Series(month_means, name = 'MonthMean', index = months)

            month_std = pd.Series(month_std, name = 'MonthMean', index = months)

            monthly_ts = (monthly_ts - month_means[month_arr].values) / month_std[month_arr].values
            
            precip_dict[name + '_precip'] = monthly_ts.dropna()
    
        
        else:
            precip_dict[name + '_precip'] = ts
            
    return precip_dict


def get_flood_ts():
    file = 'envirodata/FloodCities4.csv'
    df = pd.read_csv(file)
    df.Date = df.Date.str.slice(start = 2)
    flood_dict = {}
    for name in df.Name.drop_duplicates():
        filtered_df = df[df.Name == name]
        ts = filtered_df.FracFlooded
        ts.index = pd.to_datetime(filtered_df.Date, format = '%Y_%m_%d')
        monthly_ts = ts.resample('MS').max() 
        
        flood_dict[name + '_flood'] = monthly_ts
        
    return flood_dict

#file is csv file, monthy_dev is whether or not to aggregate by monthly z-score deviation from monthly_mean
#    enviro_type is string with either 'flood','precip', etc for column name
def get_enviro_ts(file, column_name , monthly_dev = True, split_date = False, date_format = '%Y_%m_%d' ):
    df = pd.read_csv(file)
#    print(df)
    if split_date:
        df.Date = df.Date.str.slice(start = 2)
    output_dict = {}
    for name in df.Name.drop_duplicates():
        filtered_df = df[df.Name == name]
        ts = filtered_df[column_name]
        ts.index = pd.to_datetime(filtered_df.Date, format = date_format)
        monthly_ts = ts.resample('MS').max() 
        
        output_dict[name + '_' + column_name] = monthly_ts
        if monthly_dev == True:
            month_arr = monthly_ts.index.month
            months = [x+1 for x in range(12)]
            month_means = []
            month_std = []
        
            for month in months:
                month_mean = monthly_ts[month_arr == month].mean()
                month_std_dev = monthly_ts[month_arr == month].std()
                month_means.append(month_mean)
                month_std.append(month_std_dev)
            month_means = pd.Series(month_means, name = 'MonthMean', index = months)

            month_std = pd.Series(month_std, name = 'MonthMean', index = months)

            monthly_ts = (monthly_ts - month_means[month_arr].values) / month_std[month_arr].values
            
            output_dict[name +'_'+column_name] = monthly_ts.dropna()
    
        
        else:
            output_dict[name + '_'+column_name] = monthly_ts
        
    return output_dict

precip_file = 'envirodata/ChirpsMonthlySumsRiceZones.csv'
precip_dict = get_enviro_ts(precip_file, 'precip', date_format = '%m_%d_%Y', monthly_dev  = False)


#flood_dict = get_flood_ts()

#ndvi_file = 'envirodata/NDVItwoRiceZones.csv'
#ndvi_dict = get_enviro_ts(ndvi_file,  'NDVI')

    
#precip_dict = get_precip_ts(monthly_deviations = True)
#
#fao_senegal_prices, fao_border_prices, fao_international_prices = GEIWS_prices()


#bangkok_path = 'pricedata/BangkokWFPRice.csv'
#p_df , m_df = wfp_prices(bangkok_path)

#print(precip_dict['Dakar_precip'])
#plt.plot(precip_dict['Kayes_precip'])
    

#XOF = exchange_rates()    
#
#plt.plot(senegal.Kaolack, label = 'GEIWS')
#plt.plot(prices_df.Kaolack, label = 'WFP')
#plt.legend()
#plt.suptitle('Kaolack prices, GEIWS vs WFP')


#banjul_path = '/Users/Mitchell/SenegalAnalyses/codes/pricedata/WFP_2021Feb04_Gambia_FoodPricesData_LONGGRAIN.csv'
#banjul_df, banjul_dict = wfp_prices(banjul_path, minimum_size = 0, combine = True)