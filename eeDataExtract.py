#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:03:11 2021

@author: Mitchell
"""

# import pandas as pd

# from datetime import date
# import numpy as np
# from rpy2 import robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter


    

# def make_enviro_data(country, commodity, start = '2000-01-01', country_wide = False):
    
#     import ee
#     ee.Initialize()
#     '''
#     Function to create a pandas dataframe for environmental indices to pair with food price analyses
#     -------------------
#     Arguments:
#     country (str) 
#         Country for study. If 'Senegal' custom zones will be chosen based on commodity, 
#             otherwise the whole country will be used
#     commodity (str)
#         'rice'  or 'millet', this sets the zones of growing over which to study
#     start (str, YYYY-MM-DD)
#         start of computation
#     country_wide
#         override zones and create country wide environmental parameters
        
#     '''



#     # ------ Assets -------------
#     regions = ee.FeatureCollection("users/mlt2177/SenegalAssets/regions")
#     departments = ee.FeatureCollection("users/mlt2177/SenegalAssets/departments")
#     # ------ NDVI Time Series ---------------
#     terraVeg16 = ee.ImageCollection("MODIS/006/MOD13Q1")
#     aquaVeg16 = ee.ImageCollection("MODIS/006/MYD13Q1")
#     modisVeg16 = terraVeg16.select('NDVI')
#     # ------ Precip Time Series -------------
#     CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')
#     #precip = CHIRPS.select('precipitation')
#     GPM = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06")
#     precip = GPM.select('precipitation')
#     # Total potential evapotranspiration
#     petColl = ee.ImageCollection("MODIS/006/MOD16A2").select('PET')
#     # indices in GRIDMET DROUGHT: CONUS indices
#     #gridmet = ee.ImageCollection("GRIDMET/DROUGHT")
#     # terraclimate
#     terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
#     #GLDAS 2.1 
#     gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")

#     IC_dict = {'ndvi': modisVeg16, 'precipitation': precip, 'pet': petColl, 
#             'pdsi': terraclimate.select('pdsi'), 'soilmoisture':gldas.select('SoilMoi0_10cm_inst')}
            
                    


#     # Mapping Function
#     def makeTS(enviro_param, fc, reindex = True, scale = 1000, start = '2000-01-01'):
#         '''
#         Function to make environmental time series based on param and feature collection of geometries
#         -------------------
#         enviro_param (str) 
#             'NDVI' or 'precipitation'
#         fc (ee.FeatureCollection)
#             Feature Collection of regions over which to derive time series
#         reindex (boolean), default: True
#             whether or not to aggregate data monthly
#         '''
#         start = pd.Timestamp(start)
#         end = pd.Timestamp(date.today().year, date.today().month, 1)
        
        
#         try:
#             imageCollection = IC_dict[enviro_param.lower()]
#         except KeyError:
#             raise ValueError('{} is not a valid environmetnal parameter'.format(enviro_param))
        
#         def mappingFunction(featureObj):
        
#             geom = ee.Feature(featureObj).geometry()
#             filteredColl = imageCollection.filterDate(start,end).filterBounds(geom)#.select(enviro_param)
#             collList = filteredColl.toList(filteredColl.size())
#             print(filteredColl.size().getInfo())
        
#         def makeTSmapper(imageObj):
#             image = ee.Image(imageObj)
#             meanVal = image.reduceRegion(reducer = ee.Reducer.mean(), geometry = geom, scale = scale).values().get(0)
#             return ee.Feature(None,{'Name':ee.Feature(featureObj).get('Name'),  
#                             'Date':ee.Date(image.get('system:time_start')).format(), enviro_param :meanVal})
        
#         timeSeries = ee.FeatureCollection(collList.map(makeTSmapper)).filterMetadata(enviro_param, 'not_equals', None)
#         return timeSeries

#         featuresLi  = fc.toList(fc.size()).getInfo()
#         ts_dict = {}
#         print('--- ',enviro_param,' ---')
#         for i, feature in enumerate(featuresLi):
#             print(i+1,'/', len(featuresLi ))
#             feature = ee.Feature(feature)
#             ts = mappingFunction(feature)
#             name = feature.get('Name').getInfo()
#             print(ts.size().getInfo())
#             dates = pd.to_datetime(ts.aggregate_array('Date').getInfo(),format = '%Y-%m-%dT%H:%M:%S')
#             param = ts.aggregate_array(enviro_param).getInfo()
#             series = pd.Series(data = param, index = dates, name = name)
#     #        reindex monthly if specified
#             if reindex:
#                 index = pd.date_range(start=start, end=end, freq='MS')
#                 series = series.resample("MS").mean().reindex(index)
            
            
#             ts_dict[name] = series
            
#     #    TScoll = ee.FeatureCollection(featuresLi.map(mappingFunction)).flatten();
        
#         return ts_dict


#     def get_senegal_spei(commodity, start = pd.Timestamp('2000-01-01'),
#                             end = pd.Timestamp(date.today().year, date.today().month, 1)):
        
#         path = 'envirodata/SPEI/{}SPEI1955-April2021.csv'
#         if commodity.lower() == 'millet':
#             zones = ['Kaolack','Kaffrine', 'Fatick']
#         elif commodity.lower() == 'rice':
#             zones = ['SRV','Casamance']
#         else:
#             raise ValueError(commodity + ' is invalid commodity')
#         output_dict = {}
#         for region in zones:
#             df = pd.read_csv(path.format(region))
#             dates = pd.to_datetime(df['DATA'], format = '%b%Y')
#             spei_series = df['SPEI_1']
#             spei_series.index = dates
#             new_index = pd.date_range(start=start, end=end, freq='MS')
#             spei_series = spei_series.resample("MS").mean().reindex(new_index)
#             output_dict[region + '_spei'] = spei_series
#         return output_dict

#     def get_country_spei(country, start = pd.Timestamp('2000-01-01'),
#                             end = pd.Timestamp(date.today().year, date.today().month, 1)):
        
#         path = 'envirodata/SPEI/{}SPEI1955-April2021.csv'.format(country)
#         output_dict = {}
#         df = pd.read_csv(path)
#         dates = pd.to_datetime(df['DATA'], format = '%b%Y')
#         spei_series = df['SPEI_1']
#         spei_series.index = dates
#         new_index = pd.date_range(start=start, end=end, freq='MS')
#         spei_series = spei_series.resample("MS").mean().reindex(new_index)
#         output_dict[country + '_spei'] = spei_series
#         return output_dict
            
#     # functions are inner functions to avoid issues if GEE not installed

#     end = pd.Timestamp(date.today().year, date.today().month, 1)

#     if country.lower() == 'senegal' and country_wide == False:
        
#         spei_dict = get_senegal_spei(commodity, start, end)
        
#         scale = 1000
#         # ------- millet regions: Kaolack, Kaffrine and Fatick------
#         def map_millet(feature):
#             return feature.set('Name', feature.get('ADM1_FR'))
        
#         milletRegions = regions.filter(ee.Filter.Or(ee.Filter.eq('ADM1_FR', 'Kaolack'),
#                   ee.Filter.eq('ADM1_FR', 'Kaffrine'),ee.Filter.eq('ADM1_FR', 'Fatick'))).map(map_millet)
        
#         # ----- Rice Regions: SRV, Casamance, Oriental, Sine Saloum ------
#         SRV = departments.filter(ee.Filter.Or(ee.Filter.eq('ADM2_FR', 'Dagana'),ee.Filter.eq('ADM2_FR', 'Saint-Louis'),
#             ee.Filter.eq('ADM2_FR', 'Podor'),ee.Filter.eq('ADM2_FR', 'Matam')))
        
#         Casamance = regions.filter(ee.Filter.Or(ee.Filter.eq('ADM1_FR', 'Ziguinchor'),
#                ee.Filter.eq('ADM1_FR', 'Kolda'), ee.Filter.eq('ADM1_FR', 'Sedhiou') ))
                                                
#         Oriental = regions.filter(ee.Filter.Or( ee.Filter.eq('ADM1_FR', 'Kedougou'), ee.Filter.eq('ADM1_FR', 'Tambacounda')))
        
#         SineSaloum =  regions.filter(ee.Filter.Or( ee.Filter.eq('ADM1_FR', 'Kaolack'),ee.Filter.eq('ADM1_FR', 'Kaffrine'),ee.Filter.eq('ADM1_FR', 'Fatick')))
        
        
          
#         riceGrowingZones = ee.FeatureCollection([ee.Feature(SRV.geometry(), {'Name':'SRV'}),
#                      ee.Feature(Casamance.geometry(), {'Name':'Casamance'})]) 
#                  #   ee.Feature(Oriental.geometry().union(SineSaloum.geometry()), {'Name':'OrientalAndSineSaloum'}) ])
        
#         if commodity.lower() == 'rice':
#             fc = riceGrowingZones

#         elif commodity.lower() == 'millet':
#             fc = milletRegions

#         else:
#             raise ValueError('Invalid Commodity for Senegal')
#     else:
#         try:
#             spei_dict = get_country_spei(country)
#         except FileNotFoundError:
#             print('No SPEI found for ' , country)
#             spei_dict = {}
#         scale = 5000
#         all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
#         def add_name(feature):
#             return feature.set('Name',feature.get('country_na'))
#         study_country = all_countries.filterMetadata('country_na','equals',country).map(add_name)
#         fc= study_country
        
        
#     ndvi_dict =   {key + '_ndvi' : value for key, value in makeTS('NDVI', fc, scale = scale, start = start).items() }
#     precip_dict =  {key + '_precip' : value for key, value in makeTS('precipitation', fc, scale = scale,start = start).items() }
# #    ssm_dict =  {key + '_ssm' : value for key, value in makeTS('soilmoisture', fc, scale = scale,start = start).items() }
#     ssm_dict = {}
#     pdsi_dict =  {key + '_pdsi' : value for key, value in makeTS('pdsi', fc, scale = scale,start = start).items() }
# #    spei_dict =  {key + '_spei' : value for key, value in makeTS('spei', fc, scale = scale).items() }
# #    eddi_dict =  {key + '_eddi' : value for key, value in makeTS('eddi', fc, scale = scale).items() }
    
#     df = pd.DataFrame.from_dict( {**ndvi_dict, **precip_dict,**pdsi_dict,**ssm_dict, **spei_dict}  )
    
#     return df

# def save_enviro_data(commodity, country):
#     df = make_enviro_data(commodity)
#     df.to_csv('envirodata/{}-{}-fullenviro.csv'.format(country.lower(), commodity.lower()))
    
# def update_enviro_data():
#     pass



# ----------------------------
import pandas as pd

from datetime import date
import numpy as np
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter




def make_enviro_data(country, commodity):
    '''
    Function to create a pandas dataframe for environmental indices to pair with food price analyses
    -------------------
    Arguments:
    country (str) 
        Country for study. If 'Senegal' custom zones will be chosen based on commodity, 
            otherwise the whole country will be used
    commodity (str)
        'rice'  or 'millet', this sets the zones of growing over which to study
        
    '''
    
    
    import ee
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
    # Total potential evapotranspiration
    petColl = ee.ImageCollection("MODIS/006/MOD16A2").select('PET')
    
    
    # Mapping Function
    def makeTS(enviro_param, fc, reindex = True, scale = 1000, start = '2000-01-01'):
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
        start = pd.Timestamp(start)
        end = pd.Timestamp(date.today().year, date.today().month, 1)
        
        
        if enviro_param == 'NDVI':
            imageCollection = modisVeg16 
        elif enviro_param.lower() == 'precipitation':
            imageCollection = precip
        elif enviro_param.lower() == 'pet':
            imageCollection = petColl
        else:
            raise ValueError('{} is not a valid environmetnal parameter'.format(enviro_param))
        
        def mappingFunction(featureObj):
          
          geom = ee.Feature(featureObj).geometry()
          filteredColl = imageCollection.filterDate(start,end).filterBounds(geom).select(enviro_param)
          collList = filteredColl.toList(filteredColl.size())
          
          def makeTSmapper(imageObj):
            image = ee.Image(imageObj)
            meanVal = image.reduceRegion(reducer = ee.Reducer.mean(), geometry = geom, scale = scale).values().get(0)
            return ee.Feature(None,{'Name':ee.Feature(featureObj).get('Name'),  
                            'Date':ee.Date(image.get('system:time_start')).format(), enviro_param :meanVal})
          
          timeSeries = ee.FeatureCollection(collList.map(makeTSmapper)).filterMetadata(enviro_param, 'not_equals', None)
          return timeSeries
    
        featuresLi  = fc.toList(fc.size()).getInfo()
        ts_dict = {}
        print('--- ',enviro_param,' ---')
        for i, feature in enumerate(featuresLi):
            print(i+1,'/', len(featuresLi ))
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
    
    
    if country.lower() == 'senegal':
        scale = 1000
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
                     ee.Feature(Casamance.geometry(), {'Name':'Casamance'})]) 
                 #   ee.Feature(Oriental.geometry().union(SineSaloum.geometry()), {'Name':'OrientalAndSineSaloum'}) ])
        
        if commodity.lower() == 'rice':
            fc = riceGrowingZones
        elif commodity.lower() == 'millet':
            fc = milletRegions
        else:
            raise ValueError('Invalid Commodity for Senegal')
    else:
        scale = 5000
        all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        def add_name(feature):
            return feature.set('Name',feature.get('country_na'))
        study_country = all_countries.filterMetadata('country_na','equals',country).map(add_name)
        fc= study_country
        
        
    ndvi_dict =   {key + '_ndvi' : value for key, value in makeTS('NDVI', fc, scale = scale).items() }
    precip_dict =  {key + '_precip' : value for key, value in makeTS('precipitation', fc, scale = scale).items() }
    df = pd.DataFrame.from_dict( {**ndvi_dict, **precip_dict}  )
    
    return df

def save_enviro_data(commodity, ):
    df = make_enviro_data(commodity)
    df.to_csv('envirodata/{}-fullenviro.csv'.format(commodity.lower()))
    
def update_enviro_data():
    pass





# ----------------------------






#a = make_enviro_data('Senegal', 'Rice', start = '2007-01-01')


#-------for SPEI calculation: https://spei.csic.es/map/maps.html#months=0#month=3#year=2021
#def map_millet(feature):
#    return feature.set('Name', feature.get('ADM1_FR'))
#milletRegions = regions.filter(ee.Filter.Or(ee.Filter.eq('ADM1_FR', 'Kaolack'),
#          ee.Filter.eq('ADM1_FR', 'Kaffrine'),ee.Filter.eq('ADM1_FR', 'Fatick'))).map(map_millet)
#
## ----- Rice Regions: SRV, Casamance, Oriental, Sine Saloum ------
#SRV = departments.filter(ee.Filter.Or(ee.Filter.eq('ADM2_FR', 'Dagana'),ee.Filter.eq('ADM2_FR', 'Saint-Louis'),
#    ee.Filter.eq('ADM2_FR', 'Podor'),ee.Filter.eq('ADM2_FR', 'Matam')))
#
#Casamance = regions.filter(ee.Filter.Or(ee.Filter.eq('ADM1_FR', 'Ziguinchor'),
#       ee.Filter.eq('ADM1_FR', 'Kolda'), ee.Filter.eq('ADM1_FR', 'Sedhiou') ))
#
#riceGrowingZones = ee.FeatureCollection([ee.Feature(SRV.geometry(), {'Name':'SRV'}),
#             ee.Feature(Casamance.geometry(), {'Name':'Casamance'})]) 
#rect_bounds = np.zeros((5,5),dtype = object)
#for i, element in enumerate(milletRegions.merge(riceGrowingZones).toList(100).getInfo()):
#    feature = ee.Feature(element)
#    name = feature.get('Name').getInfo()
#    bounds = np.array(feature.geometry().bounds().getInfo()['coordinates'][0])
#    top_left = (round(bounds[:,1].max(),3), round(bounds[:,0].min(),3) )
#    bottom_right = (round(bounds[:,1].min(),3),   round(bounds[:,0].max(),3))
#    rect_bounds[i] = [name, *top_left, *bottom_right]
#    print([name, *top_left, *bottom_right])
#    
#    
#    
#bounds_df = pd.DataFrame(rect_bounds, columns = ['Name','TopLeftLat','TopLeftLon',
#                                             'BottomRightLat','BottomRightLon'])
#bounds_df.to_csv('shapedata/speiboundssenegal.csv')   
#   -----------------
    



#country = 'Senegal'
#all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
#def add_name(feature):
#            return feature.set('Name',feature.get('country_na'))
#study_country = all_countries.filterMetadata('country_na','equals',country).map(add_name)
#fc= study_country
#
#precip = makeTS('precipitation', fc, scale = 5000)
#pet = makeTS('PET', fc, scale = 5000)
#
#waterbalance = {key : precip[key][precip[key].index.union(pet[key].index)] - pet[key][precip[key].index.union(pet[key].index)]
#                      for key in np.union1d(list(precip.keys()), list(pet.keys()))}
#test_waterbalance = waterbalance['Senegal'].dropna()
#test_wb_dates = test_waterbalance.index
#r_waterbalance = ro.vectors.FloatVector(test_waterbalance.values)
#
##waterbalance = {}
##for key in np.union1d(precip.keys(), pet.keys()):
##    union_index
#
#
#
#
#
##R_float_vec = ro.vectors.FloatVector(waterbalance['Senegal'].values)
#
#
#def spei_extract(r_vector):
#    path = 'SPEI/R/spei.R'
#    r=ro.r
#    r.source(path)
#    p=r.spei(r_vector, 1)
#    return p
#
#r_spei = spei_extract(r_waterbalance )

    


#enviro_param = 'NDVI'
#TScoll = makeTS(enviro_param, riceGrowingZones, reindex = True)
#dates = TScoll.aggregate_array('Date').getInfo()
#param = TScoll.aggregate_array(enviro_param).getInfo()
#region = TScoll.aggregate_array('Name').getInfo()

#df = pd.DataFrame.from_dict({'Dates':})



#print( makeTS('NDVI', riceGrowingZones).getInfo() )