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
import geopandas as gp

from extractdata import extract_giews, get_attribute
from eeDataExtract import make_enviro_data





    
#study_markets =  ['Dakar', 'Saint-Louis', 'Dagana','Nouakchott','Kayes','Tambacounda','Touba','Bakel',
#            'Banjul','Farafenni', 'Zigiunchor','Kolda', 'Basse Santa su', 'Diaobe', 'Bisseau','Conakry', 'Kaolack', 'Bangkok','Mumbai','SãoPaulo']

s,e = pd.Timestamp(2007,1,1) , pd.Timestamp(2020,3,31)
minimum_size = 160
##rice_dataframe = get_rice_df(fao_mkts_dict, None, minimum_size, s, e)
#
#
## -----------import millet---------
#senegal_millet_file = 'pricedata/SenegalGEIWSMillet.csv'
#millet_prices = GEIWS_prices(senegal_millet_file)
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


    
def plot_map(link_matrix,  names, variable, save= False):
    #upload coordinates
    
    try:    
        coord_df = pd.read_csv('shapedata/citycoordinates.csv')
        country_gdf = gp.read_file('shapedata/WestAfricaADMN0/wca_adm0.shp')
    except FileNotFoundError:
        coord_df = pd.read_csv('SenegalCodes/shapedata/citycoordinates.csv')
        country_gdf = gp.read_file('SenegalCodes/shapedata/WestAfricaADMN0/wca_adm0.shp')
        
    
    
    coord_df.index = coord_df['City Name']
    coord_df = coord_df[['Lat','Lon']]
    
    
    
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
    edit_dict = {'Casamance_ndvi' : 'NDVI',
                'Casamance_precip':'precip',
                'SRV_ndvi':'NDVI',
                'SRV_precip' : 'precip'}
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
        
        
    
def create_data(commodity):
    if commodity.lower() == 'rice' :
        senegal_rice = extract_giews(country = 'Senegal', commodity = 'rice', min_size = minimum_size)
        thai_rice = extract_giews(country = 'Thailand', commodity = 'rice', min_size = minimum_size)
        india_rice = {'Mumbai': extract_giews(country = 'India', commodity = 'rice', min_size = minimum_size)['Mumbai']}
        brazil_rice = extract_giews(country = 'Brazil', commodity = 'rice', min_size = minimum_size)
          
        mauri_rice = extract_giews(country = 'Mauritania', commodity = 'rice', min_size = minimum_size)
        guinea_rice = extract_giews(country = 'Guinea', commodity = 'rice', min_size = minimum_size)
        mali_rice = extract_giews(country = 'Guinea', commodity = 'rice', min_size = minimum_size)
        
        all_rice_dict = {**senegal_rice, **thai_rice, **india_rice}
        all_rice_dict = {**senegal_rice, **thai_rice, **india_rice, **brazil_rice, **mauri_rice, **guinea_rice, **mali_rice}  
        data = pd.DataFrame.from_dict(all_rice_dict)
        return data
        
    elif commodity.lower() == 'millet' :
        data = pd.DataFrame.from_dict(extract_giews(country = 'Senegal', commodity = 'millet', min_size = minimum_size))
        return data
    else:
        raise ValueError('Invalid Commodity')
    

    

#allows for saving of environemntal data over mulitple runs
if 'enviro_data_dict' not in dir():
    enviro_data_dict = {} 

def run_test(commodity, FDR_bool, min_lag, max_lag, add_enviro, alpha, m_y_conditioning = True, 
             interpolate = False, max_gap = 3, stationarity_method = 'firstdifference' , print_info = False, use_gee = True):
    
    
    study_data = create_data(commodity)
    
    if add_enviro:
        
        if commodity in enviro_data_dict.keys():
            enviro_df = enviro_data_dict[commodity]
        else:
            if use_gee:
                enviro_df =  -  make_enviro_data(commodity) 
            else:
                enviro_df = pd.read_csv('envirodata/{}-fullenviro.csv'.format(commodity.lower()) )
                enviro_df.index = pd.to_datetime(enviro_df.iloc[:,0], format = '%Y-%m-%d')
                enviro_df = - enviro_df.drop(columns = enviro_df.columns[0])
            enviro_data_dict[commodity] = enviro_df
        
    
    # --- Select Season -----
    
    # lean_season = [5,6,7,8,9,10]
    # harvest_season = [1,2,3,4,11,12]
    # month_mask = filter_months(adjusted_study_data, harvest_season, missing_flag = mssng)
    
#   if using month and year conditions in the system
    if m_y_conditioning: 
        m_y_data = study_data.copy()[s:e]
    #        if adding environmental variables
        if add_enviro:
            m_y_data = pd.concat([m_y_data, enviro_df], axis = 1)
            enviro_indices = [m_y_data.columns.get_loc(x) for x in enviro_df.columns ]
            
        m_y_data['Month'] = m_y_data.index.month
        m_y_data['Year'] = m_y_data.index.year
        m_y_indices = [m_y_data.columns.get_loc('Month'), m_y_data.columns.get_loc('Year')]
        m_y_data = m_y_data.interpolate(method='linear', limit=max_gap) if interpolate == True else  m_y_data
        adjusted_study_data = m_y_data
        
    else:
        if stationarity_method == 'firstdifference':
            stationary = take_first_diff
        elif stationarity_method == 'rollingmean':
            stationary = subtract_rolling_mean
        else:
            raise ValueError("Not Valid Stationarity method: 'firstdifference' or 'rollingmean'")
#            interpolate if desired
        if add_enviro: # make dataframe for NDVI and Precip over regions of commodity growth
            
            study_data = pd.concat([study_data.copy(), enviro_df], axis = 1)
            enviro_indices = [study_data.columns.get_loc(x) for x in enviro_df.columns ]
        study_data = study_data.interpolate(method='linear', limit=max_gap) if interpolate == True else  study_data
        adjusted_study_data  = stationary(adjust_seasonality(study_data.copy()))[s:e]
    
        
#        fill data with missing flag
    mssng = 99999
    filled_data = adjusted_study_data.fillna(mssng)
        
    global t
    t = filled_data.copy()
#        create tigramite dataframe
    dataframe = pp.DataFrame(filled_data.values, var_names= filled_data.columns, missing_flag = mssng)
    
#   Possible test types
    parcorr = ParCorr(significance='analytic')
    gpdc = GPDC(significance='analytic', gp_params=None)
#    select test of choice and set up pcmci
    test_type = parcorr
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=parcorr,
        verbosity=1,
        print_info = print_info)
#    get results of pcmci
    results = pcmci.run_pcmci(tau_min = min_lag, tau_max=max_lag, pc_alpha=None, no_parents = enviro_indices, month_year_indices = m_y_indices)
    #
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
    #
    if print_info:
        pcmci.print_significant_links(
                p_matrix = results['p_matrix'], 
                q_matrix = q_matrix,
                val_matrix = results['val_matrix'],
                alpha_level = alpha)
        
    pq_matrix = q_matrix if FDR_bool == True else results['p_matrix']
    link_matrix = pcmci.return_significant_positive_links(pq_matrix = pq_matrix,
                            val_matrix=results['val_matrix'], alpha_level=alpha)['link_matrix']
    link_matrix = link_matrix[:-2,:-2,:]if m_y_conditioning == True else link_matrix
    
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=filled_data.columns,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI'
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


    num = []
    caused_by = []
    causes = []
    mci = []

    for i in range(all_tau_link.shape[0]):
        for j in range(all_tau_link.shape[1]):
            icausesj = all_tau_link[i,j]
            mci_val = results['val_matrix'][i,j]
            i_name = names[i]
            j_name = names[j]
            if icausesj and i_name != j_name and True:
                print(names[i],' causes ' , names[j])
                caused_by.append(i_name)
                causes.append(j_name)
                mci.append(np.max(mci_val))
                G.add_edge(i , j)
                G.nodes[i]['influenced_by'] += 1
                n_connections +=1 
                
    link_df = pd.DataFrame.from_dict({'Caused By': caused_by,
                                     'Causes':causes,
                                     'MCI-val':mci})
                
    print('\n\n ### Causation Links ###', link_df,'\n\n')
    scale_factor = 200
    f, ax = plt.subplots(1,1,figsize = (7,5))
    f.suptitle('{} Price Causation Network'.format(commodity), fontsize = 15 )
    # ax.set_title('Arrow represents causation, circle size represents relative importance of market')
#    pos = nx.spring_layout(G)
    global pos
    
    pos = nx.drawing.layout.circular_layout(G)
    tig_pos = {'x': np.array([a[0] for a in pos.values()]) ,
               'y': np.array([a[1] for a in pos.values()]) }
    
    
    tp.plot_graph(
        val_matrix=results['val_matrix'],
        link_matrix=link_matrix,
        var_names=filled_data.columns,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI'
#        node_pos = tig_pos
        )
    plt.show()
    
    influenced_arr = scale_factor * (np.array([G.nodes[i]['influenced_by'] for i in range(len(G.nodes))]) + 1)
    label_dict = {i : G.nodes[i]['name'] for i in range(len(G.nodes)) }
    nx.draw(G, node_size = influenced_arr, with_labels = False, pos = pos, arrowsize = 30, alpha = 0.65, edge_color = 'grey', ax = ax)
    nx.draw_networkx_labels(G, pos = pos, labels = label_dict)
#    print(n_connections , ' Connections') 
    #G = nx.Graph()
#    print('shapes', np.array(names).shape, link_matrix.shape)
    plot_map(link_matrix, names, commodity,  save= False)



#commodity = 'Rice'
#FDR_bool = False
#min_lag, max_lag  = 1,4
#add_enviro = True
#alpha = 0.05
#m_y_conditioning = True 
#interpolate = False
#max_gap= 2
#stationarity_method = 'firstdifference'
#print_info = False
#
#run_test(commodity, FDR_bool, min_lag, max_lag, add_enviro, alpha, m_y_conditioning = m_y_conditioning, interpolate = interpolate,
#         max_gap= max_gap, stationarity_method = 'firstdifference', print_info = False)
#


