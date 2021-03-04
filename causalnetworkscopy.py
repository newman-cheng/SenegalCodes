#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:36:14 2021

@author: Mitchell
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#plt.style.use('ggplot')
import sklearn
# import geopandas as gp
import matplotlib.cm as cm

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

from pandas.plotting import lag_plot

from import_files import wfp_prices, GEIWS_prices, exchange_rates, get_ndvi_ts, get_precip_ts, get_flood_ts, get_enviro_ts
import itertools
import networkx as nx
import networkx.drawing.layout as lyt



from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
import pickle
from statsmodels.tsa.stattools import grangercausalitytests
import math 
from shapely.geometry import Point
import adjustText as aT

#factorial function
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

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


senegal_file = 'pricedata/SenegalRice10MarketsPrices2007to2020.csv'
border_file = 'pricedata/SurroundingMarketsSenegalPrices.csv'
international_file = 'pricedata/GEIWS_international_rice.csv'
fao_senegal_rice_prices, fao_border_rice_prices, fao_international_rice_prices = GEIWS_prices(senegal_file),GEIWS_prices(border_file), GEIWS_prices(international_file)
for df in fao_senegal_rice_prices, fao_border_rice_prices, fao_international_rice_prices:
    for column in df.columns:
        series = df[column]
        mkts_dict[column] = series
#        if column not in list(mkts_dict.keys()):
#            
#        else:
##            average previous and new series
#            mkts_dict[column] = pd.concat((mkts_dict[column], series), axis = 1).mean(axis=1,skipna = True)
#            
#        
        

plt.plot(fao_senegal_rice_prices.Dakar, label = 'fao')
plt.plot(senegal_mkts_dict['Dakar'].dropna(), label = 'WFP')
plt.legend()
plt.show() 

fao_market_sample =  ['Dakar', 'Saint-Louis', 'Dagana','Nouakchott','Kayes','Tambacounda','Touba','Bakel',
                     'Banjul','Farafenni', 'Zigiunchor','Kolda', 'Basse Santa su', 'Diaobe', 'Bisseau','Conakry', 'Kaolack']

fao_market_countries = ['Senegal', 'Senegal', 'Senegal','Mauritania','Mali','Senegal','Senegal','Senegal',
                       'Gambia','Gambia','Senegal', 'Senegal','Gambia','Senegal','Guinea Bissau','Guinea','Senegal']
tuples = list(zip(*[fao_market_countries , fao_market_sample]))
mIndex = pd.MultiIndex.from_tuples(tuples, names=["Country", "Market"])


#senegal_shapes_path  = '/Users/Mitchell/SenegalAnalyses/codes/shapedata/senegal_location_dataset/senegal_location.shp'
#senegal_shapes = gp.read_file(senegal_shapes_path)

#missing = []
#for place in fao_market_sample:
#    bool_val = place in list(mkts_dict.keys())
#    print(place,': ', bool_val)
#    if bool_val == False:
#        missing.append(place)
#print('missing: ', missing)


#------tests-----------

#test for ADF - null hypothesis that a unit root is present in a time series sample
#- null hypothesis that a unit root is present in a time series sample
def test_adfuller(arr,to_print = False):
    X1 = np.array(arr)
    X1 = X1[~np.isnan(X1)]
    
    result = adfuller(X1)
    if to_print:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        
    adf_stat, p_val = result[0], result[1]
    return adf_stat, p_val

#kpss test
def kpss_test(series, to_print = False, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    if to_print:
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critial Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')
        print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    return p_value

#-----------------------
    



#returns dict with stationarity of each arr

def check_mkts_dict(mkts_dict):
    stationary_li = []
    mkt_names = list(mkts_dict.keys())
    for mkt in mkt_names:
        print(mkt)
        mkt_data = mkts_dict[mkt].dropna()
        shift_data = mkt_data.copy()
        shift_data = (shift_data - shift_data.shift(1)).dropna()
        
        # start with assumption that each time series is stationary and valid
        stationary_valid = True
        adf_stat, p_val = test_adfuller(shift_data)
        kpss_p = kpss_test(shift_data)
        print(p_val, kpss_p)
        if p_val >= 0.05 or kpss_p < 0.05:
            stationary_valid = False
            print('test')
#            del mkts_dict[mkt]
        stationary_li.append(stationary_valid)
    return stationary_li

def grangers_causation_matrix(data, variables,maxlag, test='ssr_chi2test', verbose=False, ):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
    
#calculate the granger causality of two time series
def calc_granger_two_ts(ts1, ts2, maxlag):
    ts1 = ts1 - ts1.shift(1)
    ts2 = ts2 - ts2.shift(1)
    test_df = pd.concat((ts1, ts2), axis = 1).dropna()
    test_df.columns = ['First_ts', 'Second_ts']
    o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag)
    return o

#returns dataframe with name of markets, number of markets it causes, and visa versa
def calculate_mkt_causalities(mkts_dict, maxlag= 11, min_size = None):
    global lengths
    lengths = []
#    make empty dataframe to fill into
    mkt_names = list(mkts_dict.keys())  
    causal_df = pd.DataFrame.from_dict({"markets caused" : np.empty(len(mkt_names), dtype = str),
                                        "causes": np.zeros(len(mkt_names)),
                                        "is_caused_by": np.zeros(len(mkt_names))})
    causal_df.index = mkt_names
    
#    present_df = pd.DataFrame.from_dict({'Market': mkt_names , 
#                                         'Number of times a market causes another': np.zeros(len(mkt_names)),
#                                         'Number of times the market is caused by another':np.zeros(len(mkt_names)})
    
    
    mkt_combinations = itertools.combinations(mkts_dict.keys(), 2)
    for i, mkt_tuple in enumerate(mkt_combinations):
        if i % 10 == 0:
            print(i,'/',nCr(len(mkts_dict.keys()) , 2))
#        global first
#        global second
        first, second = mkt_tuple
        global test_df
        global first_ts
        first_ts = mkts_dict[first] - mkts_dict[first].shift(1)
        second_ts = mkts_dict[second] - mkts_dict[second].shift(1)
        
        test_df = pd.concat((first_ts, second_ts), axis = 1).dropna()
        test_df.columns = [first, second]
#        test_df = wfp_dataframe[list(mkt_tuple)].dropna()
#        shift both to remove ACF
    
#        test_df.iloc[:,0] = test_df.iloc[:,0] - test_df.iloc[:,0].shift(1)
#        test_df.iloc[:,1] = test_df.iloc[:,1] - test_df.iloc[:,1].shift(1)
#        test_df = test_df.dropna()
        try:
            o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag)
        except ValueError as e:
            print(e, mkt_tuple)
            try:
                o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag - 2)
            except ValueError as e2:
                print('second error', e2, mkt_tuple)
                continue
                    
        first_causes_second = o.iloc[1,0]
        lengths.append(len(test_df.dropna()))
        
        if len(test_df.dropna()) >= min_size:
            if first_causes_second < 0.05:
                causal_df.loc[first, 'causes'] += 1
                causal_df.loc[second,'is_caused_by'] +=1
                causal_df.loc[first, 'markets caused'] += second + ', '
                print(len(test_df.dropna()))
                
                
                
            second_causes_first = o.iloc[0,1]
            if second_causes_first < 0.05:
                causal_df.loc[second, 'causes'] += 1
                causal_df.loc[first,'is_caused_by'] +=1
                causal_df.loc[second, 'markets caused'] += first + ', '
                print(len(test_df.dropna()))
                
            
    return causal_df
            
def calculate_ndvi_causalities(ndvi_dict, mkts_dict, maxlag = 5):
#    list of how many markets ndvi causes
    ndvi_causes = np.zeros(len(ndvi_dict.keys()))
    for i, key in enumerate(ndvi_dict.keys()):
        print(i , '/' , len(ndvi_dict.keys()))
        for mkt in mkts_dict.keys():
     
            ndvi_ts = ndvi_dict[key].dropna().resample('MS').mean() 
            shift_ndvi =  ndvi_ts - ndvi_ts.shift(1)
            
            mkt_ts = mkts_dict[mkt].dropna()
            shift_mkt = mkt_ts - mkt_ts.shift(1)
            
            test_df = pd.concat((shift_ndvi, shift_mkt), axis = 1).dropna()
        
            test_df.columns = [key, mkt]
#            shift time series
#            test_df.iloc[:,0] = test_df.iloc[:,0] - test_df.iloc[:,0].shift(1)
#            test_df.iloc[:,1] = test_df.iloc[:,1] - test_df.iloc[:,1].shift(1)
            test_df = test_df.dropna()
            try:
                o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag)
            except ValueError as e:
                print(e, key)
                try:
                    o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag - 2)
                except ValueError as e2:
                    print('second error', e2, key)
                    global a
                    a = test_df
                    continue
            first_causes_second = o.iloc[1,0]
            if first_causes_second < 0.05:
                print(key, ' causes ', mkt , ' prices')
                ndvi_causes[i] += 1
    causal_series = pd.Series(  ndvi_causes, name = 'Number Market Causes')
    causal_series.index = list(ndvi_dict.keys())
    return causal_series

def calculate_precip_causalities(precip_dict, mkts_dict, maxlag = 5):
#    list of how many markets ndvi causes
    precip_causes = np.zeros(len(precip_dict.keys()))
    for i, key in enumerate(precip_dict.keys()):
        print(i , '/' , len(precip_dict.keys()))
        for mkt in mkts_dict.keys():
     
            precip_ts = precip_dict[key]
            
            mkt_ts = mkts_dict[mkt]
            shift_mkt = mkt_ts - mkt_ts.shift(1)
            
            test_df = pd.concat((precip_ts, shift_mkt), axis = 1).dropna()
            test_df.columns = [key, mkt]

            try:
                o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag)
            except ValueError as e:
                print(e, key)
                try:
                    o = grangers_causation_matrix(test_df, variables = test_df.columns, maxlag = maxlag - 2)
                except ValueError as e2:
                    print('second error', e2, key)
                    global a
                    a = test_df
                    continue
            first_causes_second = o.iloc[1,0]
            if first_causes_second < 0.05:
                print(key, ' causes ', mkt , ' prices')
                precip_causes[i] += 1
    causal_series = pd.Series(  precip_causes, name = 'Number Market Causes')
    causal_series.index = list(precip_dict.keys())
    return causal_series





#df1.groupby([pd.Grouper(freq='M'), 'status']).sum()
                
        
            
            
            
            
        
#        dictionary with markets from FAO diagram
sample_dict = { key : mkts_dict[key] for key in fao_market_sample}   







#----markets with enough observations for network analysis
#markets to checK: 
study_markets =  ['Dakar', 'Saint-Louis', 'Dagana','Nouakchott','Kayes','Tambacounda','Touba','Bakel',
                     'Banjul','Farafenni', 'Zigiunchor','Kolda', 'Basse Santa su', 'Diaobe', 'Bisseau','Conakry', 'Kaolack', 'Bangkok','Mumbai','SãoPaulo']
s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
size_th = 150
sample_dict2 = {x : mkts_dict[x] for x in study_markets if len(mkts_dict[x][s:e].dropna()) >= size_th}
#add environmental data:
ndvi_file = 'envirodata/NDVItwoRiceZones.csv'
ndvi_dict =  get_enviro_ts(ndvi_file, 'NDVI',split_date = 'True')
precip_file = 'envirodata/ChirpsMonthlySumsRiceZones.csv'
print('------')
precip_dict = get_enviro_ts(precip_file, 'precip', date_format = '%m_%d_%Y')
#add to dictionary
sample_dict2.update(ndvi_dict)
sample_dict2.update(precip_dict)


#create dataframe with only those with enoguh observations
sample2_dataframe = pd.concat( list(sample_dict2.values()), axis = 1)
sample2_dataframe.columns = list(sample_dict2.keys())



mean_size  = np.mean([len(x.dropna()) for x in sample_dict.values()])

stationarity_li = check_mkts_dict(sample_dict)
print(stationarity_li)

#ndvi_dict = get_ndvi_ts()
#flood_dict = get_flood_ts()
#precip_dict = get_precip_ts(monthly_deviations = True)
  






def map_causations(causal_df, save= False):
    #upload coordinates
    coord_df = pd.read_csv('/Users/Mitchell/SenegalAnalyses/codes/shapedata/citycoordinates.csv')
    coord_df.index = coord_df['City Name']
    coord_df = coord_df[['Lat','Lon']]
    causal_df = pd.concat((causal_df, coord_df), axis = 1)
    geometry = [Point(xy) for xy in zip(causal_df.Lon, causal_df.Lat)]
    gdf = gp.GeoDataFrame(causal_df, crs="EPSG:4326", geometry=geometry)
    
    country_gdf = gp.read_file('/Users/Mitchell/SenegalAnalyses/codes/shapedata/WestAfricaADMN0/wca_adm0.shp')
    country_gdf = country_gdf.to_crs('epsg:4326')
    country_gdf.index = country_gdf['admin0Name']
    select_country_idx = ['Senegal','Gambia','Guinea','Guinea Bissau','Mauritania','Mali']
    select_countries = country_gdf.loc[select_country_idx]
    
    #    plot map
    fig1, axs = plt.subplots(1,2,figsize = (20,15))
    
    
    vmin, vmax = 0.0, 0.3
    cmap = 'Reds'
    fig1.subplots_adjust(hspace=0.01, wspace=0.025)
    ax2, ax3  = axs
    for ax in axs:
        ax.set_ylim(8,19)
        ax.set_xlim(-18,-10)
    ax2.axis('off')
    ax3.axis('off')
    
    #blue: 607dab
    #grey: '#bfbfbf'
    select_countries.plot(ax = ax2,  facecolor='#bfbfbf', edgecolor="black")
    select_countries.plot(ax = ax3, facecolor='#bfbfbf', edgecolor="black")
                   
    matplotlib.rcParams['font.size'] = 12
    gdf.plot(ax= ax2, cmap = 'Reds', column='causes', legend= False, vmin = vmin,
             vmax = vmax, markersize = 1500*(gdf['causes']**1.5), zorder = 4)
    #    legend_kwds = {'label': "Spearman Correlation",'orientation':'vertical'})
    gdf.plot(ax= ax3, cmap = 'Reds', column='is_caused_by', legend= False, vmin = vmin,
             vmax = vmax, markersize = 1500*(gdf['is_caused_by']**1.5), zorder = 4)
    
    for ax in axs:
        texts = []
        for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.index):
            texts.append(ax.text(x + 0.15, y - 0.25, label, fontsize = 8, backgroundcolor = '#dedede', fontweight = 'bold'))

    ax2.set_title("Granger Causes %")
    ax3.set_title("Is Granger Caused By %")
    
    #    patch_col = axs[0].collections[0]
    #    pts = ax2.scatter(x_data, y_data, marker='s', c=data[x_data, y_data])
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    cmap = 'Reds'
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(gdf['causes'])
    #    p = cm.ScalarMappable( norm = norm, cmap='Reds')
    #    cb = fig2.colorbar(p, ax = axs, shrink=0.5)
    
    fig1.colorbar(m, ax = [ax2, ax3], orientation = 'horizontal', label = "%" ,shrink = 0.5, pad = 0.01)
    
    #plt.tightlayout()
    if save:
        plt.savefig('figures/GrangerCausalityMapLag{}.png'.format(maxlag), dpi = 200, bbox_inches = 'tight')
    plt.show()
    
    
def map_ndvi_causations(causal_series, variable, save= False):
    #upload coordinates
    coord_df = pd.read_csv('/Users/Mitchell/SenegalAnalyses/codes/shapedata/citycoordinates.csv')
    coord_df.index = coord_df['City Name']
    coord_df = coord_df[['Lat','Lon']]
    causal_series.index = causal_series.index.str.split('_',1, expand =True).get_level_values(0)
    causal_df = pd.concat( (causal_series, coord_df), axis = 1)
    geometry = [Point(xy) for xy in zip(causal_df.Lon, causal_df.Lat)]
    global gdf
    gdf = gp.GeoDataFrame(causal_df, crs="EPSG:4326", geometry=geometry)
    
    country_gdf = gp.read_file('/Users/Mitchell/SenegalAnalyses/codes/shapedata/WestAfricaADMN0/wca_adm0.shp')
    country_gdf = country_gdf.to_crs('epsg:4326')
    country_gdf.index = country_gdf['admin0Name']
    select_country_idx = ['Senegal','Gambia','Guinea','Guinea Bissau','Mauritania','Mali']
    select_countries = country_gdf.loc[select_country_idx]
    
    #    plot map
    fig1, ax1 = plt.subplots(1,1,figsize = (12,15))
    
    
    vmin, vmax = 0.0, 0.25
    cmap = 'Reds'
    
    
    ax1.set_ylim(8,19)
    ax1.set_xlim(-18,-10)
    ax1.axis('off')
    
    #blue: 607dab
    #grey: '#bfbfbf'
    select_countries.plot(ax = ax1,  facecolor='#bfbfbf', edgecolor="black")
                   
    matplotlib.rcParams['font.size'] = 12
    gdf.plot(ax= ax1, cmap = 'Reds', column='Number Market Causes', legend= False, vmin = vmin,
             vmax = vmax, markersize =  700 * (gdf['Number Market Causes'] / vmax), zorder = 4)
   
    
    
    texts = []
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.index):
        texts.append(ax1.text(x + 0.15, y - 0.25, label, fontsize = 8, backgroundcolor = '#dedede', fontweight = 'bold'))

    ax1.set_title(variable + " causes % markets")

    
    #    patch_col = axs[0].collections[0]
    #    pts = ax2.scatter(x_data, y_data, marker='s', c=data[x_data, y_data])
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    cmap = 'Reds'
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(gdf['Number Market Causes'])
    #    p = cm.ScalarMappable( norm = norm, cmap='Reds')
    #    cb = fig2.colorbar(p, ax = axs, shrink=0.5)
    
    fig1.colorbar(m, ax = ax1, orientation = 'horizontal', label = "%" ,shrink = 0.5, pad = 0.01)
    
    #plt.tightlayout()
    if save:
        plt.savefig('figures/NDVIGrangerCausalityMapLag{}.png'.format(maxlag), dpi = 200, bbox_inches = 'tight')
    plt.show()
        
    
#---------Market Causalities------
#maxlag = 2
#causal_df = calculate_mkt_causalities(sample_dict, maxlag = maxlag, min_size = 80)
#causal_df.index = mIndex
#causal_df = causal_df.sort_index(level = 0, ascending = False)
#causal_df['causes'] = causal_df['causes'] / len(causal_df.iloc[:,0]) 
#causal_df['is_caused_by'] = causal_df['is_caused_by'] / len(causal_df.iloc[:,0])   

#print(causal_df['causes'].sort_values(ascending=False))
#print(causal_df['is_caused_by'].sort_values(ascending=False))

#causal_df.to_csv('results/GrangerCausalitiesChartmaxlag2minsize80.csv')
#map_causations(causal_df, save= False)




#---------NDVI Causalities------

#maxlag = 2
#causal_series = calculate_ndvi_causalities(ndvi_dict, sample_dict, maxlag = maxlag)
#causal_series = causal_series / len(causal_series)  
#print(causal_series.sort_values(ascending = False))
#map_ndvi_causations(causal_series,'NDVI(lag 2)' , save= False)


#---------Precip Causalities------
#maxlag = 2
#causal_series = calculate_precip_causalities(precip_dict, sample_dict, maxlag = maxlag)
#causal_series = causal_series / len(causal_series)  
#print(causal_series.sort_values(ascending = False))
#map_ndvi_causations(causal_series, 'Precip (lag 2)' , save= False)

#---------Flood Causalities------
#maxlag = 2
#causal_series = calculate_ndvi_causalities(flood_dict, sample_dict, maxlag = maxlag)
#causal_series = causal_series / len(causal_series)  
#print(causal_series.sort_values(ascending = False))
#map_ndvi_causations(causal_series, 'Flood (lag 2)' , save= False)





#for column in wfp_dataframe.columns:
#    plt.plot(wfp_dataframe[column].dropna())
#    plt.show()

#test for lags causing nonstationarity

#
#mkt1 = 'Dakar'
#mkt2 = 'Castors'
#
#
## start with assumption that each time series is stationary and valid
#stationary_valid = True
#raw_data =  wfp_dataframe[[mkt1,mkt2]].dropna()
#lag_plot(raw_data)
#plt.show()
#test_data = raw_data.copy(deep=True)
#test_data[mkt1] = (test_data[mkt1] - test_data[mkt1].shift(1)).dropna()
#test_data[mkt2] = (test_data[mkt2] - test_data[mkt2].shift(1)).dropna()
#
#lag_plot(test_data)
#plt.show()


#msk = np.random.rand(len(test_data)) < 0.8
#train = test_data[msk].dropna()
#test = test_data[~msk].dropna()



    

# null hypothesis for both tests
#test_bools = [False, False, False, False]
#for i, mkt in enumerate((mkt1, mkt2)):
#    arr = test_data[mkt].dropna()
#    adf_stat, p_val = test_adfuller(arr)
#    if p_val < 0.05:
#        test_bools[2*i] = True
#    print(test_data[mkt])
#    kpss_p = kpss_test(arr)
#    if kpss_p >= 0.05:
#        test_bools[2*i + 1] = True
#    if np.isnan(kpss_p) or np.isnan(p_val):
#        raise ValueError
#        
#if False not in test_bools:
#    model = VAR(raw_data.dropna()) #recall that rawData is w/o difference operation
#    lags = [1,2,3,4,5,6,7,8,9,10,11,12]
#    aics = []
#    bics = []
#    fpes = []
#    hqics = []
#    for i in lags:
#        result = model.fit(i)
#        aics.append(result.aic)
#        bics.append(result.bic)
#        fpes.append(result.fpe)
#        hqics.append(result.hqic)
#        
#        try:
#            print('Lag Order =', i)
#            print('AIC : ', result.aic)
#            print('BIC : ', result.bic)
#            print('FPE : ', result.fpe)
#            print('HQIC: ', result.hqic, '\n')
#        except:
#            continue
#
#plt.plot(lags, aics)
#plt.show()
#plt.plot(lags, bics)
#plt.show()
#plt.plot(lags, fpes)
#plt.show()
#plt.plot(lags, hqics)
#plt.show()
            
        
        
#model = VAR(train)
#model_fitted = model.fit(lags[aics.index(min(aics))])


# if close to 2, no significant serial correlation
#from statsmodels.stats.stattools import durbin_watson
#out = durbin_watson(model_fitted.resid)
#
#for col, val in zip(test_data.columns, out):
#    print(col, ':', round(val, 2))
#
#
#
#cointegrated = False
#import statsmodels.tsa.stattools as ts 
#result=ts.coint(test_data[mkt1].dropna(), test_data[mkt2].dropna())
#print(result)
#p_val = result[1]
#if p_val < 0.05:
#    cointegrated = True




#o = grangers_causation_matrix(train, variables = train.columns)  
#print(o)
    
s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
test_df = sample2_dataframe.copy()[s:e]
t = test_df.copy()
#        global test_df
#    whether or not to add environmental variables to analysis
add_enviro = False
#number of environmental indices added to end of df
enviro_index = 4


#subtract by rolling mean
def subtract_rolling_mean(df, window_size = 3):
    rolling_mean = df.rolling(window_size).mean()
    adjust_df = df - rolling_mean
    return adjust_df

# handle seasonality by subtracting monthly mean
def adjust_seasonality(df):
    for x in range(len(df.columns)):
        m = df.iloc[:,x].index.month
        mon_avg = []
        months = [i + 1 for i in range(12)]
        for mon in months:
            filt = df.iloc[:,x][m == mon]
            avg = filt.mean()
            mon_avg.append(avg)
        month_series = pd.Series(mon_avg, index = months)[m]
        df.iloc[:,x] = df.iloc[:,x] - month_series.values
    return df
    
    

#for x in range(len(test_df.columns[:-enviro_index])):
#    test_df.iloc[:,x] = test_df.iloc[:,x] - test_df.iloc[:,x].shift(1)
#remove environemtal variables if applicable
clip_index = -enviro_index if add_enviro == False else None
mssng = 99999
test_df = test_df.iloc[: , : clip_index].copy().fillna(mssng)
t2 = test_df.copy()

# import millet
senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
millet_prices = GEIWS_prices(senegal_millet_file)
millet_prices = subtract_rolling_mean( adjust_seasonality(millet_prices ) )



study_data = millet_prices

# give custom NAN value for tigramite to interpret
mssng = 99999
study_data = study_data.copy().fillna(mssng)



    
dataframe = pp.DataFrame(study_data.values, var_names= study_data.columns, missing_flag = mssng)
tp.plot_timeseries(dataframe)
parcorr = ParCorr(significance='analytic')

gpdc = GPDC(significance='analytic', gp_params=None)

pcmci_gpdc = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=gpdc,
    verbosity=0)

pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=1)
#
min_lag, max_lag  = 1,6
results = pcmci.run_pcmci(tau_min = min_lag, tau_max=max_lag, pc_alpha=None)
#
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
#
pcmci.print_significant_links(
        p_matrix = results['p_matrix'], 
        q_matrix = q_matrix,
        val_matrix = results['val_matrix'],
        alpha_level = 0.05)

link_matrix = pcmci.return_significant_links(pq_matrix = results['p_matrix'],
                        val_matrix=results['val_matrix'], alpha_level=0.05)['link_matrix']
tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=study_data.columns,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    )
plt.show()

#dataframe = dataframe.iloc[:,:-4]
G = nx.DiGraph()
for i , node_name in enumerate(dataframe.var_names):
#    G.add_node((i,{'name':node_name}))
    G.add_node(i,name = node_name, influenced_by = 0)
    
    
#make N^2 matrix
n_connections = 0
all_tau_link = np.max(link_matrix, axis = 2)
for i in range(all_tau_link.shape[0]):
    for j in range(all_tau_link.shape[1]):
        icausesj = all_tau_link[i,j]
        if icausesj:
            i_name = dataframe.var_names[i]
            j_name = dataframe.var_names[j]
            print(dataframe.var_names[i],' causes ' , dataframe.var_names[j])
            G.add_edge(i , j)
            G.nodes[i]['influenced_by'] += 1
            n_connections +=1 
            
scale_factor = 100
pos = nx.spring_layout(G)
c = nx.drawing.layout.circular_layout(G)
influenced_arr = scale_factor * (np.array([G.nodes[i]['influenced_by'] for i in range(len(G.nodes))]) + 1)
label_dict = {i : G.nodes[i]['name'] for i in range(len(G.nodes)) }
nx.draw(G, node_size = influenced_arr, with_labels = True, labels = label_dict, pos = c)
print(n_connections , ' Connections')
#G = nx.Graph()


