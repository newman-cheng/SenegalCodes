#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:00:51 2021

@author: Mitchell
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from tigramitecustom import data_processing as pp
from tigramitecustom import plotting as tp
from tigramitecustom.pcmci import PCMCI
from tigramitecustom.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import matplotlib.cm as cm
import networkx as nx
import networkx.drawing.layout as lyt
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot

from import_files import wfp_prices, GEIWS_prices, exchange_rates, get_ndvi_ts, get_precip_ts, get_flood_ts, get_enviro_ts
import geopandas as gp

def get_rice_dict():
    # RICE price imports
    #----- prices from WFP data, senegal markets
    wfp_imported_rice_path = '/Users/Mitchell/SenegalAnalyses/SenegalCodes/pricedata/WFP_Senegal_importedrice.csv'
    senegal_wfp_dataframe, senegal_mkts_dict = wfp_prices(wfp_imported_rice_path, minimum_size = 0)
    mkts_dict = senegal_mkts_dict.copy()
    
    #WFP farafenni, basse, gambia:
    
    gambia_path = '/Users/Mitchell/SenegalAnalyses/SenegalCodes/pricedata/WFPFarafenniBasseLongGrain.csv'
    gambia_df, gambia_dict = wfp_prices(gambia_path, minimum_size = 0, combine = True)
    del gambia_dict['CombinedAverage']
    for key in gambia_dict.keys():
        mkts_dict[key] = gambia_dict[key]
    
    #---- WFP, banjul, Gambia
    banjul_path = '/Users/Mitchell/SenegalAnalyses/SenegalCodes/pricedata/WFP_2021Feb04_Gambia_FoodPricesData_LONGGRAIN.csv'
    banjul_df, banjul_dict = wfp_prices(banjul_path, minimum_size = 0, combine = True)
    mkts_dict['Banjul'] = banjul_dict['CombinedAverage']
    
    bissau_path = '/Users/Mitchell/SenegalAnalyses/SenegalCodes/pricedata/WFP_2021Feb04_Guinea-Bissau_FoodPricesData.csv'
    bissau_df, bissau_dict = wfp_prices(bissau_path, minimum_size = 0, combine = True)
    mkts_dict['Bisseau'] = bissau_dict['CombinedAverage']
    
    
    
    conakry_path = '/Users/Mitchell/SenegalAnalyses/SenegalCodes/pricedata/GuineaConakryAllMarkets.csv'
    conakry_df, conakry_dict = wfp_prices(conakry_path, minimum_size = 0, combine = True)
    mkts_dict['Conakry'] = bissau_dict['CombinedAverage']
    
    
    senegal_file = 'pricedata/SenegalRice10MarketsPrices2007to2020.csv'
    border_file = 'pricedata/SurroundingMarketsSenegalPrices.csv'
    international_file = 'pricedata/GEIWS_international_rice.csv'
    fao_senegal_rice_prices, fao_border_rice_prices, fao_international_rice_prices = GEIWS_prices(senegal_file),GEIWS_prices(border_file), GEIWS_prices(international_file)
    fao_mkts_dict = {}
    for df in fao_senegal_rice_prices, fao_border_rice_prices, fao_international_rice_prices:
        for column in df.columns:
            series = df[column]
            mkts_dict[column] = series
            fao_mkts_dict[column] = series
            
    return mkts_dict, fao_mkts_dict

def get_enviro_df(monthly_dev = True):
    ndvi_file = 'envirodata/NDVItwoRiceZones.csv'
    ndvi_dict =  get_enviro_ts(ndvi_file, 'NDVI',split_date = 'True', monthly_dev = monthly_dev)
    precip_file = 'envirodata/ChirpsMonthlySumsRiceZones.csv'
    precip_dict = get_enviro_ts(precip_file, 'precip', date_format = '%m_%d_%Y', monthly_dev  = monthly_dev)
    
#    enviro_dict =  {**ndvi_dict, **precip_dict}
#    
#    enviro_df = pd.DataFrame.from_dict(enviro_dict)
    enviro_df = pd.concat(list(ndvi_dict.values()) + list(precip_dict.values()), axis = 1)
    enviro_df.columns = list(ndvi_dict.keys()) + list(precip_dict.keys())
    return enviro_df


mkts_dict, fao_mkts_dict = get_rice_dict()
    
study_markets =  ['Dakar', 'Saint-Louis', 'Dagana','Nouakchott','Kayes','Tambacounda','Touba','Bakel',
            'Banjul','Farafenni', 'Zigiunchor','Kolda', 'Basse Santa su', 'Diaobe', 'Bisseau','Conakry', 'Kaolack', 'Bangkok','Mumbai','SãoPaulo']
s,e = pd.Timestamp(2000,1,1) , pd.Timestamp(2020,12,31)
def get_rice_df(mkts_dict, study_markets, min_size , s,e ):
    # dict with only fao GEIWS markets
    sample_dict = {x : mkts_dict[x] for x in mkts_dict.keys() if 
                       len(mkts_dict[x][s:e].dropna()) >= min_size }
    
    rice_dict = sample_dict
    
    rice_dataframe = pd.concat( list(rice_dict.values()), axis = 1)
    rice_dataframe.columns = list(rice_dict.keys())
    
    return rice_dataframe
minimum_size = 160
rice_dataframe = get_rice_df(fao_mkts_dict, None, minimum_size, s, e)


# -----------import millet---------
senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
millet_prices = GEIWS_prices(senegal_millet_file)
#  -------------------------------




#subtract by rolling mean
def subtract_rolling_mean(df, window_radius = 6):
    rolling_mean = df.rolling(window_radius*2 , center = True).mean(skipna = True)
    adjust_df = df - rolling_mean
    return adjust_df
# first differences over dataframe
def take_first_diff(df):
    if type(df) != pd.core.frame.DataFrame: 
        df = pd.DataFrame(df)
    for x in range(len(df.columns)):
        df.iloc[:,x] = df.iloc[:,x] - df.iloc[:,x].shift(1)
    return df

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
def check_stationarity(df):
    for i in range(df.shape[1]):
        ts = df.iloc[:,i].dropna()
        name = df.columns[i]
#       adf test and kpss test
        adf_stat, p_val = test_adfuller(ts)
        # kpss_p = kpss_test(ts)
        kpss_p = 1000
        print(name, p_val)
        
        # print(p_val, kpss_p)
        if p_val >= 0.05 or kpss_p < 0.05:
            print('{} non stationary due to ADF and KPSS'.format(name))
        elif p_val >= 0.05:
            print('{} non stationary due to ADF'.format(name))
        elif kpss_p < 0.05:
            print('{} non stationary due to KPSS'.format(name))


# filter by months to include in time series allowing for seasonality corrections
# pass df with datetime index and list/array of month numbers (1-12) to include 
# def filter_months(df, month_arr, missing_flag = None):
#     m = df.index.month
#     bool_arr = np.isin(m , month_arr)
#     if missing_flag  == None:
#         missing_flag = np.nan
#     df[~bool_arr] = missing_flag
#     return df

# filter by months to include in time series allowing for seasonality corrections
# pass df with datetime index and list/array of month numbers (1-12) to include 
def filter_months(df, month_arr, missing_flag = None):
    m = df.index.month
    bool_arr = np.isin(m , month_arr)
    # if missing_flag  == None:
    #     missing_flag = np.nan
    # df[~bool_arr] = missing_flag
    month_mask_arr = ~bool_arr
    month_mask = np.repeat(np.expand_dims(month_mask_arr, axis=1), repeats=df.shape[1], axis=1)
    return month_mask
    
def fit_distribution(dataframe):
    dist_names = ['weibull_min','norm','weibull_max','beta','invgauss',
                  'uniform','gamma','expon','lognorm','pearson3','triang']
    
    
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data 
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)
    
    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        print("{}\n{}\n".format(dist, param))
    
    
        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
    
        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)
    
    
    #Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)
    
    
    print ('\nDistributions listed by Betterment of fit:')
    print ('............................................')
    print (results)
    
    
def plot_map(link_matrix,  names, variable, save= False):
    #upload coordinates
    global coord_df
    coord_df = pd.read_csv('/Users/Mitchell/SenegalAnalyses/SenegalCodes/shapedata/citycoordinates.csv')
    coord_df.index = coord_df['City Name']
    coord_df = coord_df[['Lat','Lon']]
    
    
    country_gdf = gp.read_file('/Users/Mitchell/SenegalAnalyses/SenegalCodes/shapedata/WestAfricaADMN0/wca_adm0.shp')
    country_gdf = country_gdf.to_crs('epsg:4326')
    country_gdf.index = country_gdf['admin0Name']
    select_country_idx = ['Senegal','Gambia','Guinea','Guinea Bissau','Mauritania','Mali']
    select_countries = country_gdf.loc[select_country_idx]
    
    #    plot map
    fig1, ax1 = plt.subplots(1,1,figsize = (12,15))
    
    
    vmin, vmax = 0.0, 0.25
    cmap = 'Reds'
    
    #blue: 607dab
    #grey: '#bfbfbf'
    select_countries.plot(ax = ax1,  facecolor='#bfbfbf', edgecolor="black")
                   
    matplotlib.rcParams['font.size'] = 12
    global  G
    G = nx.DiGraph()
    edit_dict = {'SouhternRainfedArea_NDVI' : 'NDVI',
                'SouhternRainfedArea_precip':'precip',
                'NorthernRiverValley_NDVI':'NDVI',
                'NorthernRiverValley_precip' : 'precip'}
#    names = [edit_dict[name] if name in edit_dict.keys() else name for name in names]
    for i , node_name in enumerate(names):
    #    G.add_node((i,{'name':node_name}))
        G.add_node(i,name = node_name, influenced_by = 0)
        
        
    #make N^2 matrix
    n_connections = 0
    all_tau_link = np.max(link_matrix, axis = 2)
    for i in range(all_tau_link.shape[0]):
        for j in range(all_tau_link.shape[1]):
            icausesj = all_tau_link[i,j]
            i_name = names[i]
            j_name = names[j]
            if icausesj and i_name != j_name:
#                print(names[i],' causes ' , names[j])
                G.add_edge(i , j)
                G.nodes[i]['influenced_by'] += 1
                n_connections +=1 
                
    scale_factor = 200
    position_dict = {}
    for i in range(len(names)):   
        name = names[i].replace('ã','a')
        lon, lat = coord_df.loc[name].Lon, coord_df.loc[name].Lat
        position_dict[i] = np.array([lon, lat])
       

    lons, lats = [float(val[0]) for val in position_dict.values()] , [float(val[1]) for val in position_dict.values()]
    buffer = 1
    min_lon, max_lon = min(lons) - buffer, max(lons) + buffer
    min_lat, max_lat = min(lats) - buffer,  max(lats) + buffer
    ax1.set_ylim(min_lat, max_lat)
    ax1.set_xlim(min_lon, max_lon)
    ax1.axis('on')
        
    # ax.set_title('Arrow represents causation, circle size represents relative importance of market')

#    pos = nx.spring_layout(G)
#    c = nx.drawing.layout.circular_layout(G)
    
    influenced_arr = scale_factor * (np.array([G.nodes[i]['influenced_by'] for i in range(len(G.nodes))]) + 1) - 100
    print(influenced_arr)
    label_dict = {i : G.nodes[i]['name'] for i in range(len(G.nodes)) }
#    change enviro vars to better names:
    
    nx.draw(G, node_size = influenced_arr , with_labels = False, pos = position_dict, arrowsize = 30, alpha = 0.85,  ax = ax1)
#    nx.draw_networkx_labels(G, pos = position_dict, labels = label_dict)
#    print(n_connections , ' Connections')
    #G = nx.Graph()
  
   
    
    
    texts = []
    for x, y, label in zip(lons, lats, names):
        texts.append(ax1.text(x , y , label, fontsize = 14, ))# fontweight = 'bold'))

    ax1.set_title(variable + " Causation Map")

    
    #    patch_col = axs[0].collections[0]
    #    pts = ax2.scatter(x_data, y_data, marker='s', c=data[x_data, y_data])
#    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
#    cmap = 'Reds'
#    m = cm.ScalarMappable(norm=norm, cmap=cmap)
#    m.set_array(gdf['Number Market Causes'])
    #    p = cm.ScalarMappable( norm = norm, cmap='Reds')
    #    cb = fig2.colorbar(p, ax = axs, shrink=0.5)
    
#    fig1.colorbar(m, ax = ax1, orientation = 'horizontal', label = "%" ,shrink = 0.5, pad = 0.01)
    
    #plt.tightlayout()
    if save:
        plt.savefig('figures/TigramiteMap{}.png'.format(variable), dpi = 200, bbox_inches = 'tight')
    plt.show()
        
    
def test_distribution(dataframe, t = None):
    def print_res(p, alpha):
        print('p = ', p)
        if np.isnan(p):
            print('p is null')
        elif p < alpha: 
            print("The null hypothesis of normality can be rejected --> NOT NORMAL")
        else:
            print("The null hypothesis of normality cannot be rejected --> LIKELY NORMAL")
    alpha = 0.05
    global arr
    arr = dataframe.values.flatten()
    arr = arr[~np.isnan(arr)]
    corrected = (arr - np.mean(arr)) / np.std(arr)
    plt.hist(corrected, bins = 15)
    plt.suptitle(t)
    plt.show()
    qqplot(corrected, line = '45')
    plt.suptitle(t + ' qq Plot')
    plt.show()
    # test raw values
    print("Raw Data:")
    k2, p = normaltest(corrected) 
    print_res(p, alpha)
    
    # print('Log Data')
    # log_arr = np.log(arr)
    # k2, p = normaltest(log_arr)
    # print_res(p, alpha)
    
    
# test_distribution(rice_dataframe , t='Rice')
# test_distribution(millet_prices, t = 'Millet')
# test_distribution()


# for i, x in enumerate([ rice_dataframe,   subtract_rolling_mean( adjust_seasonality(rice_dataframe.copy())), 
#           millet_prices,  subtract_rolling_mean( adjust_seasonality(millet_prices.copy()))]):
#     print(i)

#     test_distribution(x)
#     print('\n\n\n')

        
    

    



def run_test(commodity, FDR_bool, min_lag, max_lag, add_enviro, alpha, m_y_conditioning = False):
    # ------------------OPTIONS-------------
    # --select data for study
    if commodity.lower() == 'millet':
        study_data = mam_millet_dataframe
#        study_data = millet_prices
    elif commodity.lower() == 'rice':
        study_data = rice_dataframe
    # study_data = millet_prices
    # --whether or not to run BH FDR algorithm reducing false positives
    # FDR_bool = True
    # --mimimum and maximum lag to look for 
    # min_lag, max_lag  = 1,4
    # --add environmental variables
    # add_enviro = False
    # ------------------------------
    
    # give custom NAN value for tigramite to , and adjust for seasonality and take rolling mean
    mssng = 99999

#    adjusted_study_data  = subtract_rolling_mean( study_data.copy())[s:e]
    adjusted_study_data  = subtract_rolling_mean( adjust_seasonality(study_data.copy()))[s:e]
#    adjusted_study_data  = take_first_diff( adjust_seasonality(study_data.copy()))[s:e]
    
    global t
    t = adjusted_study_data
    
    
    check_stationarity(adjusted_study_data)
    global filled_data
    filled_data = adjusted_study_data.fillna(mssng)
    
    
    
    # seasonal Adjustment  
    
    # lean_season = [5,6,7,8,9,10]
    # harvest_season = [1,2,3,4,11,12]
    # month_mask = filter_months(adjusted_study_data, harvest_season, missing_flag = mssng)
    enviro_indices = None
    if add_enviro:
        enviro_df = get_enviro_df()[s:e].fillna(mssng)
        filled_data = pd.concat([filled_data, enviro_df], axis = 1)
        enviro_indices = [filled_data.columns.get_loc(x) for x in enviro_df.columns ]
    

#   if using month and year conditions in the system
    m_y_indices = None
    if m_y_conditioning:
        min_lag = 0
        filled_data = study_data.copy()[s:e]
#        if adding environmental variables
        if add_enviro:
            enviro_df = get_enviro_df(monthly_dev = False)[s:e]
            filled_data = pd.concat([filled_data, enviro_df], axis = 1)
            enviro_indices = [filled_data.columns.get_loc(x) for x in enviro_df.columns ]
        filled_data['Month'] = filled_data.index.month
        filled_data['Year'] = filled_data.index.year
        m_y_indices = [filled_data.columns.get_loc('Month'), filled_data.columns.get_loc('Year')]
        filled_data = filled_data.fillna(mssng)
    
    
        
    dataframe = pp.DataFrame(filled_data.values, var_names= filled_data.columns, missing_flag = mssng)
    # tp.plot_timeseries(dataframe)
    parcorr = ParCorr(significance='analytic')
    
#    gpdc = GPDC(significance='analytic', gp_params=None)
    
    # pcmci_gpdc = PCMCI(
    #     dataframe=dataframe, 
    #     cond_ind_test=gpdc,
    #     verbosity=0)
    global pcmci
    global results
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=parcorr,
        verbosity=1)
    #
    
    results = pcmci.run_pcmci(tau_min = min_lag, tau_max=max_lag, pc_alpha=None, no_parents = enviro_indices, month_year_indices = m_y_indices)
    #
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
    #
    pcmci.print_significant_links(
            p_matrix = results['p_matrix'], 
            q_matrix = q_matrix,
            val_matrix = results['val_matrix'],
            alpha_level = alpha)
    pq_matrix = q_matrix if FDR_bool == True else results['p_matrix']
    link_matrix = pcmci.return_significant_links(pq_matrix = pq_matrix,
                            val_matrix=results['val_matrix'], alpha_level=alpha)['link_matrix']
    link_matrix = link_matrix[:-2,:-2,:]if m_y_conditioning == True else link_matrix
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=filled_data.columns,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        )
    plt.show()
    
    #dataframe = dataframe.iloc[:,:-4]
    G = nx.DiGraph()
    names = dataframe.var_names[:-2] if m_y_conditioning == True else dataframe.var_names
    for i , node_name in enumerate(names):
    #    G.add_node((i,{'name':node_name}))
        G.add_node(i,name = node_name, influenced_by = 0)
        
        
    #make N^2 matrix
    n_connections = 0
#    remove contemp if tau min= 0 
    all_tau_link = np.max(link_matrix, axis = 2)
#    all_tau_link = np.max(link_matrix, axis = 2)[:-2,:-2] if m_y_conditioning == True else np.max(link_matrix, axis = 2)

    for i in range(all_tau_link.shape[0]):
        for j in range(all_tau_link.shape[1]):
            icausesj = all_tau_link[i,j]
            i_name = names[i]
            j_name = names[j]
            if icausesj and i_name != j_name and True:
                print(names[i],' causes ' , names[j])
                G.add_edge(i , j)
                G.nodes[i]['influenced_by'] += 1
                n_connections +=1 
                
    scale_factor = 200
    f, ax = plt.subplots(1,1,figsize = (7,5))
    f.suptitle('{} Price Causation Network'.format(commodity), fontsize = 15 )
    # ax.set_title('Arrow represents causation, circle size represents relative importance of market')
#    pos = nx.spring_layout(G)
    pos = nx.drawing.layout.circular_layout(G)
    influenced_arr = scale_factor * (np.array([G.nodes[i]['influenced_by'] for i in range(len(G.nodes))]) + 1)
    label_dict = {i : G.nodes[i]['name'] for i in range(len(G.nodes)) }
    nx.draw(G, node_size = influenced_arr, with_labels = False, pos = pos, arrowsize = 30, alpha = 0.65, edge_color = 'grey', ax = ax)
    nx.draw_networkx_labels(G, pos = pos, labels = label_dict)
    print(n_connections , ' Connections')
    #G = nx.Graph()
#    print('shapes', np.array(names).shape, link_matrix.shape)
    plot_map(link_matrix, names, commodity,  save= False)

#----STL-----
#from statsmodels.tsa.seasonal import STL
#stl = STL(co2, seasonal=13)
#res = stl.fit()
#fig = res.plot()

commodity = 'Rice'
FDR_bool = False
min_lag, max_lag  = 1,4
add_enviro = True
alpha = 0.001
m_y_conditioning = False

#run_test(commodity, FDR_bool, min_lag, max_lag, add_enviro, alpha, m_y_conditioning = m_y_conditioning)



