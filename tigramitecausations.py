#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:00:51 2021

@author: Mitchell
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import networkx as nx
import networkx.drawing.layout as lyt

from pandas.plotting import lag_plot

from import_files import wfp_prices, GEIWS_prices, exchange_rates, get_ndvi_ts, get_precip_ts, get_flood_ts, get_enviro_ts



# RICE price imports
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

    
s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,12,31)
minimum_size = 140
sample_dict = {x : mkts_dict[x] for x in mkts_dict.keys() if len(mkts_dict[x][s:e].dropna()) >= minimum_size}

rice_dataframe = pd.concat( list(sample_dict.values()), axis = 1)
rice_dataframe.columns = list(sample_dict.keys())



# -----------import millet---------
senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
millet_prices = GEIWS_prices(senegal_millet_file)
#  -------------------------------

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
    




# ------------------OPTIONS-------------
# select data for study
study_data = rice_dataframe
# whether or not to run BH FDR algorithm reducing false positives
FDR_bool = True
# mimimum and maximum lag to look for 
min_lag, max_lag  = 1,3
# ------------------------------

# give custom NAN value for tigramite to , and adjust for seasonality and take rolling mean
mssng = 99999
adjusted_study_data  = adjust_seasonality( subtract_rolling_mean(study_data.copy()) )[s:e]
t = adjusted_study_data.copy()
adjusted_study_data.fillna(mssng)




    
dataframe = pp.DataFrame(adjusted_study_data.values, var_names= adjusted_study_data.columns, missing_flag = mssng)
# tp.plot_timeseries(dataframe)
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

results = pcmci.run_pcmci(tau_min = min_lag, tau_max=max_lag, pc_alpha=None)
#
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
#
pcmci.print_significant_links(
        p_matrix = results['p_matrix'], 
        q_matrix = q_matrix,
        val_matrix = results['val_matrix'],
        alpha_level = 0.05)

pq_matrix = q_matrix if FDR_bool == True else results['p_matrix']
link_matrix = pcmci.return_significant_links(pq_matrix = pq_matrix,
                        val_matrix=results['val_matrix'], alpha_level=0.05)['link_matrix']
tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=adjusted_study_data.columns,
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

