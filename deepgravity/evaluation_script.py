import json
import pandas as pd
# import geopandas as gpd
import zipfile
import torch

import deepgravity.od_models as od
import deepgravity.ffnn as ffnn
import deepgravity.evaluate_models as ev
# import od_models as od
# import ffnn as ffnn
# import evaluate_models as ev


# paths

wk_dir = '/Users/fs13378/sda3/projects/deep_gravity/'

uk = {
    'region_name': 'uk',
    'db_dir': wk_dir + 'data/uk/'
}

trapani = {
    'region_name': 'trapani',
    'db_dir': wk_dir + 'data/trapani/'
}

sassari = {
    'region_name': 'sassari',
    'db_dir': wk_dir + 'data/sassari/'
}

padova = {
    'region_name': 'padova',
    'db_dir': wk_dir + 'data/padova/'
}


def load_data(info_dict):

    db_dir = info_dict['db_dir']

    # osm features
    archive = zipfile.ZipFile(db_dir+'tileid2oa2handmade_features.json.zip', 'r')
    with archive.open('tileid2oa2handmade_features.json') as f:
    # with open(db_dir+'tileid2oa2handmade_features.json.zip', 'r') as f:
        tileid2oa2features2vals = json.load(f)

    # geographic info
    oa_gdf = pd.read_csv(db_dir+'oa_gdf.csv.gz', dtype={'geo_code': 'str'})

    # flows
    flow_df = pd.read_csv(db_dir+'flows_oa.csv.zip', dtype={'residence': 'str', 'workplace': 'str'})

    # coarse grained geography
    msoa_df = pd.read_csv(db_dir+'msoa_df_all.csv.zip', dtype={'oa11cd': 'str', 'msoa11cd': 'str'}).set_index('oa11cd')

    oa2msoa = msoa_df.to_dict()['msoa11cd']
    # oa2lsoa = msoa_df.to_dict()['lsoa11cd']

    msoa2oa = {}
    for k,v in oa2msoa.items():
        try:
            msoa2oa[v] += [k]
        except KeyError:
            msoa2oa[v]  = [k]

    return tileid2oa2features2vals, oa_gdf, flow_df, oa2msoa, msoa2oa


def load_ffnn_model(tileid2oa2features2vals, oa2features, oa2centroid):

    if torch.cuda.is_available():
        # run on GPU
        device = torch.device("cuda:0")
    else:
        # run on CPU
        device = torch.device("cpu")

    lr = 5e-6
    momentum = 0.9
    dim_hidden = 256
    dropout_p = 0.0
    dim_origins = 100
    model_args = [lr, momentum, dim_hidden, dropout_p, dim_origins]

    params = '_'.join(map(str, [lr, momentum, dim_hidden, dropout_p, dim_origins]))
    fname = wk_dir + 'data/model_UK_commute_colab_NOarea_%s.pt' % params
    # fname = wk_dir + 'data/model_UK_commute.pt'

    loc_id = list(tileid2oa2features2vals[next(iter(tileid2oa2features2vals))].keys())[0]
    dim_w = len(ffnn.get_features_ffnn(loc_id, loc_id, oa2features, oa2centroid))  # trX.size()[2]
    model = ffnn.NN_MultinomialRegression(dim_w, dim_hidden, dropout_p=dropout_p, device=device)

    checkpoint = torch.load(fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # train_losses = checkpoint['train_losses']
    # test_losses = checkpoint['test_losses']
    # cpc_trains = checkpoint['cpc_trains']
    # cpc_tests = checkpoint['cpc_tests']

    if 'cuda' in device.type:
        print('using GPU')
        model.cuda()
        print(model.device)

    # uncomment line below to perform inference
    modelstatus = model.eval()
    # uncomment line below to keep training
    # modelstatus = model.train()

    return model, optimizer, model_args


def load_original_model(tileid2oa2features2vals, oa2pop, oa2centroid):

    if torch.cuda.is_available():
        # run on GPU
        device = torch.device("cuda:0")
    else:
        # run on CPU
        device = torch.device("cpu")

    lr = 5e-6
    momentum = 0.9
    df = 'exponential'
    dim_origins = 100
    model_args = [df, lr, momentum, dim_origins]

    params = '_'.join(map(str, [df, lr, momentum, dim_origins]))
    fname = wk_dir + 'data/original_gravity_UK_commute_%s.pt' % params

    loc_id = list(tileid2oa2features2vals[next(iter(tileid2oa2features2vals))].keys())[0]
    dim_w = len(od.get_features_original_gravity(loc_id, loc_id, oa2pop, oa2centroid))  # trX.size()[2]
    model = od.NN_OriginalGravity(dim_w, df=df, device=device)

    checkpoint = torch.load(fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # train_losses = checkpoint['train_losses']
    # test_losses = checkpoint['test_losses']
    # cpc_trains = checkpoint['cpc_trains']
    # cpc_tests = checkpoint['cpc_tests']

    if 'cuda' in device.type:
        print('using GPU')
        model.cuda()
        print(model.device)

    # uncomment line below to perform inference
    modelstatus = model.eval()
    # uncomment line below to keep training
    # modelstatus = model.train()

    return model, optimizer, model_args


def evaluate(info_dict, model_name='deepgravity'):
    db_dir = info_dict['db_dir']

    # tileid2oa2features2vals, oa_gdf, flow_df, oa2msoa, msoa2oa, oa_pop = load_uk_data()
    tileid2oa2features2vals, oa_gdf, flow_df, oa2msoa, msoa2oa = load_data(info_dict)

    # evaluate on ALL tiles in tileid2oa2features2vals
    train_tiles = list(tileid2oa2features2vals.keys())

    oa2features = {}
    oa2centroid = {}
    od2flow = {}
    oa2features, oa2centroid, od2flow, tileid2oa2features2vals, oa_gdf, flow_df = \
        ev.update_loc_dicts_with_new_tile(train_tiles[0], oa2features, oa2centroid, od2flow,
                                       tileid2oa2features2vals, oa_gdf, flow_df)

    if model_name == 'deepgravity':
        model, optimizer, model_args = load_ffnn_model(tileid2oa2features2vals, oa2features, oa2centroid)
    else:
        # Resident population
        oa_pop = flow_df.groupby('residence').sum()
        oa2pop = oa_pop.to_dict()['commuters']
        # add oa's with 0 population
        all_oas = set(oa_gdf['geo_code'].values)
        oa2pop = {**{o: 1e-6 for o in all_oas}, **oa2pop}
        model, optimizer, model_args = load_original_model(tileid2oa2features2vals, oa2pop, oa2centroid)

    tile2performance = \
        ev.evaluate_model_performance(train_tiles, model, oa2msoa,
                                      tileid2oa2features2vals, oa_gdf, flow_df,
                                      max_od=100000, min_oa_in_tile=20, return_flows=False)

    region_name = info_dict['region_name']
    with open(db_dir + 'tile2performance_NOarea_%s_%s.json' % (region_name, model_name), 'w') as f:
        json.dump(tile2performance, f)


if __name__ == '__main__':

    model_name = 'deepgravity'
    # model_name = 'original'

    evaluate(padova, model_name=model_name)
    evaluate(sassari, model_name=model_name)
    evaluate(trapani, model_name=model_name)
