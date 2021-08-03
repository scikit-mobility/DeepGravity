import random
import numpy as np
import pandas as pd
import json
import zipfile
import gzip
import pickle
import torch
import string

try:
    import od_models as od
    import ffnn
except ModuleNotFoundError:
    import deepgravity.od_models as od
    import deepgravity.ffnn as ffnn


def load_data(db_dir, tile_size='25000'):
    # tileid2oa2features2vals
    archive = zipfile.ZipFile(db_dir + tile_size + '/tileid2oa2handmade_features.json.zip', 'r')
    with archive.open('tileid2oa2handmade_features.json') as f:
        tileid2oa2features2vals = json.load(f)

    # oa_gdf
    oa_gdf = pd.read_csv(db_dir + 'oa_gdf.csv.gz', dtype={'geo_code': 'str'})

    # flow_df
    flow_df = pd.read_csv(db_dir + 'flows_oa.csv.zip', \
                          dtype={'residence': 'str', 'workplace': 'str'})

    # msoa_df
    msoa_df = pd.read_csv(db_dir + 'msoa_df_all.csv.zip', \
                          dtype={'oa11cd': 'str', 'msoa11cd': 'str'}).set_index('oa11cd')

    # oa2msoa
    oa2msoa = msoa_df.to_dict()['msoa11cd']

    # oa2pop
    oa_pop = flow_df.groupby('residence').sum()
    oa2pop = oa_pop.to_dict()['commuters']
    # add oa's with 0 population
    all_oas = set(oa_gdf['geo_code'].values)
    oa2pop = {**{o: 1e-6 for o in all_oas}, **oa2pop}

    # oa2features, od2flow, oa2centroid
    with open(db_dir + '/oa2features.pkl', 'rb') as f:
        oa2features = pickle.load(f)

    with open(db_dir + tile_size + '/od2flow.pkl', 'rb') as f:
        od2flow = pickle.load(f)

    with open(db_dir + '/oa2centroid.pkl', 'rb') as f:
        oa2centroid = pickle.load(f)

    return tileid2oa2features2vals, oa_gdf, flow_df, msoa_df, oa2msoa, oa2pop, \
        oa2features, od2flow, oa2centroid


def load_model(fname, oa2centroid, oa2features, oa2pop, device, df='exponential', \
               distances=None, dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=True):
    """
    for model of class NN_OriginalGravity, oa2features is oa2pop

    """
    # loc_id = list(tileid2oa2features2vals[next(iter(tileid2oa2features2vals))].keys())[0]
    loc_id = list(oa2centroid.keys())[0]

    if 'original_gravity' in fname or '_MFG_' in fname:
        dim_w = len(od.get_features_original_gravity(loc_id, loc_id, oa2pop, oa2centroid, df='exponential_all'))
        model = od.NN_OriginalGravity(dim_w, df=df, device=device)
        print('OG')
    elif 'original_gravity_people' in fname or '_G_' in fname:
        dim_w = len(od.get_features_original_gravity(loc_id, loc_id, oa2features, oa2centroid, df='exponential'))
        model = od.NN_OriginalGravity(dim_w, df=df, device=device)
        print('OGP')
    elif 'deepgravity_people' in fname or '_NG_' in fname:
        dim_w = len(ffnn.get_features_ffnn(loc_id, loc_id, oa2pop, oa2centroid, 'deepgravity_people', distances=None, k=0))
        model = ffnn.NN_MultinomialRegression(dim_w, dim_hidden, 'deepgravity_people', dropout_p=dropout_p, device=device)
        print('DGP')
    elif 'deepgravity_knn' in fname or '_DGknn_' in fname:
        dim_w = len(ffnn.get_features_ffnn(loc_id, loc_id, oa2features, oa2centroid, 'deepgravity_knn', k=2, distances=distances))
        model = ffnn.NN_MultinomialRegression(dim_w, dim_hidden, 'deepgravity_knn', distances=distances, k=2, dropout_p=dropout_p, device=device)
        print('DGPKNN')
    else:
        dim_w = len(ffnn.get_features_ffnn(loc_id, loc_id, oa2features, oa2centroid, 'deepgravity', distances=None, k=0)) #trX.size()[2]
        model = ffnn.NN_MultinomialRegression(dim_w, dim_hidden, 'deepgravity',  dropout_p=dropout_p, device=device)
        print('DG')
    checkpoint = torch.load(fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #
    #train_losses = checkpoint['train_losses']
    #test_losses = checkpoint['test_losses']
    #cpc_trains = checkpoint['cpc_trains']
    #cpc_tests = checkpoint['cpc_tests']
    # 
    #if 'cuda' in device.type:
    #    model.cuda()
    #    if verbose:
    #        print('using GPU')
    #        print(model.device)
    #
    #    for state in optimizer.state.values():
    #        for k, v in state.items():
    #            if torch.is_tensor(v):
    #                state[k] = v.cuda()
    # 
    ## uncomment line below to perform inference
    #modelstatus = model.eval()
    ## uncomment line below to keep training
    ## modelstatus = model.train()

    return model#, optimizer, train_losses, test_losses, cpc_trains, cpc_tests


def instantiate_model(oa2centroid, oa2features, oa2pop, dim_input, device=torch.device("cpu"), \
        df='deepgravity', dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=False, distance=None, \
        k=4):
    """
    use:
        df = 'deepgravity'  for deep gravity
        df = 'exponential'  for exponential original gravity
        df = 'powerlaw'     for power law original gravity
    """

    if df in ['deepgravity', 'DG']:
        model = ffnn.NN_MultinomialRegression(dim_input, dim_hidden,  'deepgravity', dropout_p=dropout_p, device=device)
        if verbose:
            print('DG', dim_input)
    elif df in ['deepgravity_people', 'NG']:
        model = ffnn.NN_MultinomialRegression(dim_input, dim_hidden, 'deepgravity_people', dropout_p=dropout_p, device=device)
        if verbose:
            print('DGP', dim_input)
    elif df in ['deepgravity_knn', 'DGknn']:
        model = ffnn.NN_MultinomialRegression(dim_input, dim_hidden, 'deepgravity_knn', distances=distance, k=k, dropout_p=dropout_p, device=device)
        if verbose:
            print('DGPKNN', dim_input)
    elif df in ['exponential', 'G']:
        model = od.NN_OriginalGravity(dim_input, df=df, device=device)
        if verbose:
            print('OGP', dim_input)
    elif df in ['exponential_all', 'MFG']:
        model = od.NN_OriginalGravity(dim_input, df=df, device=device)
        if verbose:
            print('OG', dim_input)

    #if 'cuda' in device.type:
    #    print('using GPU')
    #    model.cuda()

    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum) #, weight_decay=1e-2)


    #train_losses = []
    #test_losses = []
    #cpc_trains = []
    #cpc_tests = []

    return model#, optimizer, train_losses, test_losses, cpc_trains, cpc_tests


def random_string(N):
    """
    Generate a random string of length "N" 
    with  lowercase letters and integers.
    """
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(N))


def get_k_closest(l0, k, oa2centroid, oa_list):
    dist_oa = [[od.earth_distance(oa2centroid[l0], oa2centroid[l]), l]
               for l in oa_list if l != l0]
    return list(zip(*sorted(dist_oa)[:k]))[1]


def compute_oa2closestk(db_dir, k=8):
    """for each loc find k closest locs
    
    db_dir: path to file oa2centroid.
    """
    import shapely
    import geopandas as gpd

    with open(db_dir + 'oa2centroid.pkl', 'rb') as f:
        oa2centroid = pickle.load(f)

    gdf = gpd.GeoDataFrame([[l, shapely.geometry.Point([p[1], p[0]])] for l,p in oa2centroid.items()], 
                 columns=['oa', 'geometry'])
    spatial_index = gdf.sindex
    oa2closestk = {l0: get_k_closest(l0, k, oa2centroid, \
                       [gdf.iloc[r].oa for r in \
                       spatial_index.nearest(oa2centroid[l0][::-1], num_results=64)][1:])
               for l0 in list(oa2centroid.keys())}

    # save
    with gzip.open(db_dir + f'oa2closest{k}.json.zip', 'w') as fout:
        fout.write(json.dumps(oa2closestk).encode('utf-8'))  


def load_oa2closestk(db_dir, k=8):
    with gzip.open(db_dir + f'oa2closest{k}.json.zip', 'r') as fin:
        oa2closestk = json.loads(fin.read().decode('utf-8'))
    return oa2closestk


def compute_oa2featuresk(oa2closestk, oa2features, k=4):
    oa2featuresk = {oa: feats + 
                list(np.mean(np.array([oa2features[l] for l in oa2closestk[oa][:k]]), axis=0))
                    for oa,feats in oa2features.items()}
    return oa2featuresk

