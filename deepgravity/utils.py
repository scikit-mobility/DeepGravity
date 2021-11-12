import random
import numpy as np
import pandas as pd
import json
import zipfile
import gzip
import pickle
import torch
import string
import os

import geopandas
from skmob.tessellation import tilers

from math import sqrt, sin, cos, pi, asin

from importlib.machinery import SourceFileLoader

path = './models/deepgravity.py'
ffnn = SourceFileLoader('ffnn', path).load_module()

def _is_support_files_computed(db):
    if os.path.isdir(db+'/processed'):
        base = db + '/processed/'
        return os.path.isfile(base+'tileid2oa2handmade_features.json') and os.path.isfile(base+'oa_gdf.csv.gz') and os.path.isfile(base+'flows_oa.csv.zip') and os.path.isfile(base+'msoa_df_all.csv.zip') and os.path.isfile(base+'oa2features.pkl') and os.path.isfile(base+'oa2centroid.pkl')
    else:
        return False
    
def _check_base_files(db_dir):
    # tessellation look up: can be eithre tessellation.shp, tessellation.geojson 
    if not (os.path.isfile(db_dir+'/tessellation.shp') or os.path.isfile(db_dir+'/tessellation.geojson')):
        raise ValueError('Tessellation is missing! There must be a file named tessellation.shp or tessellation.geojson')
    if not (os.path.isfile(db_dir+'/output_areas.shp') or os.path.isfile(db_dir+'/output_areas.geojson')):
        raise ValueError('Output areas are missing! There must be a file named output_areas.shp or output_areas.geojson')
    if not (os.path.isfile(db_dir+'/flows.csv')):
        raise ValueError('Flows are missing! There must be a file named flows.csv')

    print('Tessellation, Flows and Output Areas have been found....')    
       
    
def _compute_support_files(db_dir, tile_id_column, tile_geometry, oa_id_column, oa_geometry, flow_origin_column, flow_destination_column, flow_flows_column):
    # first, we check if there are at least the needed files into the base directory. 
    _check_base_files(db_dir)
    print('Generating the processed files - it may take a while....')
    print('Reading tessellation....')
    try: 
        tessellation = geopandas.read_file(db_dir+'/tessellation.shp', dtype={tile_id_column:str})
    except:
        tessellation = geopandas.read_file(db_dir+'/tessellation.geojson', dtype={tile_id_column:str})
    tessellation = tessellation[[tile_id_column, tile_geometry]]
    print('Reading output areas....')
    try: 
        output_areas = geopandas.read_file(db_dir+'/output_areas.shp', dtype={oa_id_column:str})
    except:
        output_areas = geopandas.read_file(db_dir+'/output_areas.geojson', dtype={oa_id_column:str})
    output_areas = output_areas[[oa_id_column, oa_geometry]]
    print('Reading features....')
    try:
        features = pd.read_csv(db_dir+'/features.csv')
        if not oa_id_column in list(features.columns):
            raise ValueError('Features must be associated with an output area. Please add a column '++' to features.csv')
    except:
        features = None
        print('Running without features. features.csv not found....')
        
    print('Mapping output areas with tessellation....')    
    output_areas['centroid'] = output_areas[oa_geometry].centroid
    # prepare and write  oa_gdf.csv.gz
    output_areas["area_km2"] = output_areas[oa_geometry].area/ 10**6
    output_areas['x'] = output_areas['centroid'].x
    output_areas['y'] = output_areas['centroid'].y
    output_areas['ctrs'] = '[' + output_areas['x'].astype(str) + ',' + output_areas['y'].astype(str) + ']' 
    
    temp_out = output_areas[[oa_id_column, 'ctrs','area_km2']]
    temp_out.rename(columns={oa_id_column:'geo_code', 'ctrs':'centroid'},inplace=True)
    
    temp_out.to_csv(db_dir+'/processed/oa_gdf.csv.gz')
    
    oa2centroid = {}
    for i,row in temp_out.iterrows():
        oa2centroid[row['geo_code']] = row['centroid']
        
    with open(db_dir+'/processed/oa2centroid.pkl', 'wb') as handle:
        pickle.dump(oa2centroid, handle)
    
    output_areas.drop(columns=[oa_geometry], inplace=True)
    output_areas.rename(columns={'centroid':oa_geometry},inplace=True)
    
    mapping = geopandas.sjoin(output_areas, tessellation, how="inner", op="within")
    try:
        mapping.drop(columns=['index_right'],inplace=True)
    except:
        pass
    
    flows = pd.read_csv(db_dir+'/flows.csv', dtype={flow_origin_column:str, flow_destination_column:str, flow_flows_column:int})
    flows = flows[[flow_origin_column, flow_destination_column, flow_flows_column]]
    
    flows.rename(columns={flow_origin_column:'residence', flow_destination_column:'workplace', flow_flows_column:'commuters'},inplace=True)
    flows.to_csv(db_dir+'/processed/flows_oa.csv.zip')
    
    od2flow = {}
    for i,row in flows.iterrows():
        od2flow[(row['residence'],row['workplace'])] = row['commuters']
        
    with open(db_dir+'/processed/od2flow.pkl', 'wb') as handle:
        pickle.dump(oa2centroid, handle)
    
    features = pd.read_csv(db_dir+'features.csv', dtype={oa_id_column:str})
    
    oa2features = {}
    for i,row in features.iterrows():
        oa2features[row[0]]=row[1:].values

    tileid2oa2handmade_features = dict()
    for i,row in mapping.iterrows():
        if row[tile_id_column] not in tileid2oa2handmade_features:
            tileid2oa2handmade_features[row[tile_id_column]] = dict()
            tileid2oa2handmade_features[row[tile_id_column]][row[oa_id_column]]=dict()
        else:
            tileid2oa2handmade_features[row[tile_id_column]][row[oa_id_column]]=dict()
    for i,row in features.iterrows():
        for item in zip(list(row.keys()),row.values):
            tileid2oa2handmade_features[row[tile_id_column]][item[0]]=[item[1]]
    
    with open('tileid2oa2handmade_features.json', 'w') as f:
        json.dump(tileid2oa2handmade_features, f)
            
            
def tessellation_definition(db_dir,name,size):
    if not (os.path.isfile(db_dir+'/tessellation.shp') or os.path.isfile(db_dir+'/tessellation.geojson')):
        tessellation = tilers.tiler.get("squared", base_shape=name, meters=size)
        tessellation.to_file(db_dir+'/tessellation.shp')
    
def load_data(db_dir, tile_id_column, tile_geometry, oa_id_column, oa_geometry, flow_origin_column, flow_destination_column, flow_flows_column):
    # check if there are the computed information
    #if not _is_support_files_computed(db_dir):
    #    _compute_support_files(db_dir, tile_id_column, tile_geometry, oa_id_column, oa_geometry, flow_origin_column, flow_destination_column, flow_flows_column)
        
    # tileid2oa2features2vals
    with open(db_dir + '/processed/tileid2oa2handmade_features.json') as f:
        tileid2oa2features2vals = json.load(f)

    # oa_gdf
    oa_gdf = pd.read_csv(db_dir + '/processed/oa_gdf.csv.gz', dtype={'geo_code': 'str'})

    # flow_df
    flow_df = pd.read_csv(db_dir + '/processed/flows_oa.csv.zip', \
                          dtype={'residence': 'str', 'workplace': 'str'})
    
    # oa2pop
    oa_pop = flow_df.groupby('residence').sum()
    oa2pop = oa_pop.to_dict()['commuters']
    # add oa's with 0 population
    all_oas = set(oa_gdf['geo_code'].values)
    
    oa2pop = {**{o: 1e-6 for o in all_oas}, **oa2pop}

    for k in oa2pop:
        if oa2pop[k] == 0:
            oa2pop[k] = 1e-6
    
    # oa2features, od2flow, oa2centroid
    with open(db_dir + '/processed/oa2features.pkl', 'rb') as f:
        oa2features = pickle.load(f)

    with open(db_dir + '/processed/od2flow.pkl', 'rb') as f:
        od2flow = pickle.load(f)

    with open(db_dir + '/processed/oa2centroid.pkl', 'rb') as f:
        oa2centroid = pickle.load(f)

    return tileid2oa2features2vals, oa_gdf, flow_df, oa2pop, oa2features, od2flow, oa2centroid


def load_model(fname, oa2centroid, oa2features, oa2pop, device, dim_s=1, \
               distances=None, dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=True):
    loc_id = list(oa2centroid.keys())[0]

    model = ffnn.NN_MultinomialRegression(dim_s, dim_hidden, 'deepgravity',  dropout_p=dropout_p, device=device)
    checkpoint = torch.load(fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def instantiate_model(oa2centroid, oa2features, oa2pop, dim_input, device=torch.device("cpu"), dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=False):

    model = ffnn.NN_MultinomialRegression(dim_input, dim_hidden,  'deepgravity', dropout_p=dropout_p, device=device)

    return model


def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...