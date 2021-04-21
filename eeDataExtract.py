#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:03:11 2021

@author: Mitchell
"""
import numpy as np 
import pandas as pd
import ee
from datetime import date
ee.Initialize()

# ------ Assets -------------
regions = ee.FeatureCollection("users/mlt2177/SenegalAssets/regions")
departments = ee.FeatureCollection("users/mlt2177/SenegalAssets/departments")
# ------ NDVI Time Series ---------------
terraVeg16 = ee.ImageCollection("MODIS/006/MOD13Q1")
aquaVeg16 = ee.ImageCollection("MODIS/006/MYD13Q1")
modisVeg16 = terraVeg16
# ------ Precip Time Series -------------
CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
#precip = CHIRPS.select('precipitation')
GPM = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06")
precip = GPM.select('precipitation')


# ------- millet regions: Kaolack, Kaffrine and Fatick------
def map_millet(feature):
    return feature.set('Name', feature.get('ADM1_FR'))

milletRegions = regions.filter(ee.Filter.Or(ee.Filter.eq('ADM1_FR', 'Kaolack'),
          ee.Filter.eq('ADM1_FR', 'Kaffrine'),ee.Filter.eq('ADM1_FR', 'Fatick'))).map(map_millet)

# ----- Rice Regions: SRV, Casamance, Oriental, Sine Saloum ------
SRV = departments.filter(ee.Filter.Or(ee.Filter.eq('ADM2_FR', 'Dagana'),ee.Filter.eq('ADM2_FR', 'Saint-Louis'),
    ee.Filter.eq('ADM2_FR', 'Podor'),ee.Filter.eq('ADM2_FR', 'Matam')))

Casamance = regions.filter(ee.Filter.Or(ee.Filter.eq('ADM1_FR', 'Ziguinchor'),
       ee.Filter.eq('ADM1_FR', 'Kolda'), ee.Filter.eq('ADM1_FR', 'Sedhiou') ))
                                        
Oriental = regions.filter(ee.Filter.Or( ee.Filter.eq('ADM1_FR', 'Kedougou'), ee.Filter.eq('ADM1_FR', 'Tambacounda')))

SineSaloum =  regions.filter(ee.Filter.Or( ee.Filter.eq('ADM1_FR', 'Kaolack'),ee.Filter.eq('ADM1_FR', 'Kaffrine'),ee.Filter.eq('ADM1_FR', 'Fatick')))


  
riceGrowingZones = ee.FeatureCollection([ee.Feature(SRV.geometry(), {'Name':'SRV'}),
             ee.Feature(Casamance.geometry(), {'Name':'Casamance'}), 
             ee.Feature(Oriental.geometry().union(SineSaloum.geometry()), {'Name':'OrientalAndSineSaloum'}) ])




# Mapping Function
def makeTS(enviro_param, fc, reindex = True):
    '''
    Function to make environmental time series based on param and feature collection of geometries
    -------------------
    enviro_param (str) 
        'NDVI' or 'precipitation'
    fc (ee.FeatureCollection)
        Feature Collection of regions over which to derive time series
    reindex (boolean), default: True
        whether or not to aggregate data monthly
    '''
    start = pd.Timestamp('2000-01-01')
    end = pd.Timestamp(date.today().year, date.today().month, 1)
    
    
    if enviro_param == 'NDVI':
        imageCollection = modisVeg16 
    elif enviro_param.lower() == 'precipitation':
        imageCollection = precip
    else:
        raise ValueError('{} is not a valid environmetnal parameter'.format(enviro_param))
    
    def mappingFunction(featureObj):
      
      geom = ee.Feature(featureObj).geometry()
      filteredColl = imageCollection.filterDate(start,end).filterBounds(geom).select(enviro_param)
      collList = filteredColl.toList(filteredColl.size())
      
      def makeTSmapper(imageObj):
        image = ee.Image(imageObj)
        meanVal = image.reduceRegion(reducer = ee.Reducer.mean(), geometry = geom, scale = 1000).values().get(0)
        return ee.Feature(None,{'Name':ee.Feature(featureObj).get('Name'),  'Date':ee.Date(image.get('system:time_start')).format(), enviro_param :meanVal})
      
      timeSeries = ee.FeatureCollection(collList.map(makeTSmapper))
      return timeSeries

    featuresLi  = fc.toList(fc.size())
    ts_dict = {}
    for i, feature in enumerate(featuresLi.getInfo()):
        feature = ee.Feature(feature)
        ts = mappingFunction(feature)
        name = feature.get('Name').getInfo()
        dates = pd.to_datetime(ts.aggregate_array('Date').getInfo(),format = '%Y-%m-%dT%H:%M:%S')
        param = ts.aggregate_array(enviro_param).getInfo()
        series = pd.Series(data = param, index = dates, name = name)
#        reindex monthly if specified
        if reindex:
            index = pd.date_range(start=start, end=end, freq='MS')
            series = series.resample("MS").mean().reindex(index)
        
        
        ts_dict[name] = series
        
#    TScoll = ee.FeatureCollection(featuresLi.map(mappingFunction)).flatten();
    
    return ts_dict


def make_enviro_data(commodity):
    '''
    Function to create a pandas dataframe for environmental indices to pair with food price analyses
    
    
    if commodity.lower() == 

enviro_param = 'NDVI'
TScoll = makeTS(enviro_param, riceGrowingZones, reindex = True)
#dates = TScoll.aggregate_array('Date').getInfo()
#param = TScoll.aggregate_array(enviro_param).getInfo()
#region = TScoll.aggregate_array('Name').getInfo()

#df = pd.DataFrame.from_dict({'Dates':})



#print( makeTS('NDVI', riceGrowingZones).getInfo() )