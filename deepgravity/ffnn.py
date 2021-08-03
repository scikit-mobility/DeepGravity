import json
import pandas as pd
# import geopandas as gpd
import shapely
import area
import numpy as np
import random
import torch
from zipfile import ZipFile
from math import sqrt, sin, cos, pi, asin
from ast import literal_eval

try:
    import od_models as od
except ModuleNotFoundError:
    import deepgravity.od_models as od


# Paths

osm_dir = '/path/to/osm/data/'
uk_dir = '/path/to/uk/data/'


def df_to_dict(df):
    split = df.to_dict(orient='split')
    keys = split['index']
    values = split['data']
    return {k: v for k, v in zip(keys, values)}


def earth_distance(lat_lng1, lat_lng2):
    """
    Compute the distance (in km) along earth between two lat/lon pairs
    :param lat_lng1: tuple
        the first lat/lon pair
    :param lat_lng2: tuple
        the second lat/lon pair

    :return: float
        the distance along earth in km
    """
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...


def get_geom_centroid(geom, return_lat_lng=False):
    lng, lat = map(lambda x: x.pop(), geom.centroid.xy)
    if return_lat_lng:
        return [lat, lng]
    else:
        return [lng, lat]


def poly_area_km2(vertices):
    """
    Example
    -------

    Compute areas of polygons in a geodataframe

        [poly_area_km2(list(zip(*w[0].exterior.xy))) for w in gdf.way]

    """
    return area.area({"type": "Polygon", "coordinates": [vertices]}) / 1e6


def get_area_km2(ww):
    if type(ww) == shapely.geometry.polygon.Polygon:
        return poly_area_km2(list(zip(*ww.exterior.xy)))
    else:
        return sum([poly_area_km2(list(zip(*w.exterior.xy))) for w in ww])


def load_oa_features(tile_id='264', tileid2oa2features=None, oa_gdf=None, flow_df=None):

    # Features
    if tileid2oa2features is None:
        with open(osm_dir + 'oa2handmade_features/tileid2oa2handmade_features.json', 'r') as f:
            tileid2oa2features = json.load(f)

    # Select locations inside the tile_id
    oa2features = tileid2oa2features[tile_id]

    # Transform each feature in a float
    oa2feat = {k: {kk: vv[0] for kk, vv in v.items()} for k, v in oa2features.items()}
    oa_feat_df = pd.DataFrame.from_dict(oa2feat, orient='index')

    # Geometries
    if oa_gdf is None:
        print(' Reading shapefile... ')
        oa_uk_shp = uk_dir+'boundary_data/infuse_oa_lyr_2011_shp/'
        oa_gdf = gpd.read_file(oa_uk_shp)

    # Select locations inside the tile_id
    oa_features = oa_gdf[oa_gdf['geo_code'].isin(oa2features.keys())].copy()

    # Compute centroids, if not present
    if 'centroid' not in oa_features.columns:
        print(' computing centroid... ')
        centr = oa_features.geometry.apply(get_geom_centroid, return_lat_lng=True)
        oa_features.loc[:, 'centroid'] = centr
    else:
        oa_features['centroid'] = oa_features['centroid'].apply(literal_eval)

    # Compute areas, if not present
    if 'area_km2' not in oa_features.columns:
        print(' computing area... ')
        oa_features.loc[:, 'area_km2'] = oa_features['geometry'].apply(get_area_km2)

    # Merge features and geometries
    # oa_features.drop(columns=['label', 'name', 'geometry'], inplace=True)
    oa_features = oa_features.loc[:, ['geo_code', 'centroid', 'area_km2']]
    oa_features = oa_features.join(oa_feat_df, on='geo_code')
    oa_features.set_index('geo_code', inplace=True)

    # OA's IDs
    oa_within = oa_features.index.values

    # Flows
    if flow_df is None:
        flow_data_file = uk_dir+'flow_data/WF01BUK_oa_v2.zip'
        zip_file = ZipFile(flow_data_file)
        flow_df = pd.read_csv(zip_file.open('wf01buk_oa_v2.csv'), header=None,
                              names=['residence', 'workplace', 'commuters'])

    # Select only flows between the OAs within the tile_id
    flow_df_within = flow_df[(flow_df['residence'].isin(oa_within)) &
                             (flow_df['workplace'].isin(oa_within))].copy()
    flow_df_within.set_index(['residence', 'workplace'], verify_integrity=True, inplace=True)

    # Transform to dictionaries
    ## Flows
    od2flow = flow_df_within.to_dict()['commuters']


    # ## Features except centroid
    # df = oa_features.loc[:, ~oa_features.columns.isin(['centroid'])]
    # oa2features = df_to_dict(df)

    ## Features except centroid and area
    df = oa_features.loc[:, ~oa_features.columns.isin(['centroid'])]
    # divide each feature by area
    df = df.div(df['area_km2'], axis=0)
    # remove the column area
    df = df.loc[:, ~df.columns.isin(['area_km2'])]
    oa2features = df_to_dict(df)


    ## Centroids
    oa2centroid = oa_features['centroid'].to_dict()

    return oa2centroid, oa2features, od2flow, tileid2oa2features, oa_gdf, flow_df


# Compute features for train and test sets


def get_features_ffnn(oa_origin, oa_destination, oa2features, oa2centroid, df, distances, k):
    # dist_od = distance(oa2centroid[oa_origin], oa2centroid[oa_destination]).km

    if df == 'deepgravity':
        dist_od = earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
        return oa2features[oa_origin] + oa2features[oa_destination] + [dist_od]

    elif df=='deepgravity_knn':
        return oa2features[oa_origin] + oa2features[oa_destination] + distances[oa_origin] + distances[oa_destination]
    else :
        # here oa2features is oa2pop
        dist_od = earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
        return [np.log(oa2features[oa_origin])] + [np.log(oa2features[oa_destination])] + [dist_od]


def get_flow(oa_origin, oa_destination, o2d2flow):
    try:
        # return od2flow[(oa_origin, oa_destination)]
        return o2d2flow[oa_origin][oa_destination]
    except KeyError:
        return 0


def get_destinations(oa, size_train_dest, all_locs_in_train_region, o2d2flow, frac_true_dest=0.5):
    try:
        true_dests_all = list(o2d2flow[oa].keys())
    except KeyError:
        true_dests_all = []
    size_true_dests = min(int(size_train_dest * frac_true_dest), len(true_dests_all))
    size_fake_dests = size_train_dest - size_true_dests
    # print(size_train_dest, size_true_dests, size_fake_dests, len(true_dests_all))

    true_dests = np.random.choice(true_dests_all, size=size_true_dests, replace=False)
    fake_dests_all = list(set(all_locs_in_train_region) - set(true_dests))
    fake_dests = np.random.choice(fake_dests_all, size=size_fake_dests, replace=False)

    dests = np.concatenate((true_dests, fake_dests))
    np.random.shuffle(dests)
    return dests


def split_train_test_sets(oas, fraction_train):
    """
    :param oas: list
        list(tileid2oa2features[tile_id].keys())

    :param fraction_train:

    :return:

    """

    # oa_within = oa_features.index.values
    # oa_within = list(oa2features.keys())

    n = len(oas)
    dim_train = int(n * fraction_train)

    random.shuffle(oas)
    train_locs = oas[:dim_train]
    test_locs = oas[dim_train:]

    return train_locs, test_locs


class NN_MultinomialRegression(od.NN_OriginalGravity):

    def __init__(self, dim_input, dim_hidden, df, dropout_p=0.35, distances=None, k=2, device=torch.device("cpu")):
        """
        dim_input = 2
        dim_hidden = 20
        """
        super(od.NN_OriginalGravity, self).__init__(dim_input, device=device)
        # super().__init__(self)

        self.df = df
        self.distances = distances
        self.k = k

        self.device = device

        p = dropout_p

        self.linear1 = torch.nn.Linear(dim_input, dim_hidden)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p)

        self.linear2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(p)

        self.linear3 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(p)

        self.linear4 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(p)

        # self.linear_out = torch.nn.Linear(dim_hidden, 1)

        self.linear5 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu5 = torch.nn.LeakyReLU()
        self.dropout5 = torch.nn.Dropout(p)

        self.linear6 = torch.nn.Linear(dim_hidden, dim_hidden // 2)
        self.relu6 = torch.nn.LeakyReLU()
        self.dropout6 = torch.nn.Dropout(p)

        self.linear7 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu7 = torch.nn.LeakyReLU()
        self.dropout7 = torch.nn.Dropout(p)

        self.linear8 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu8 = torch.nn.LeakyReLU()
        self.dropout8 = torch.nn.Dropout(p)

        self.linear9 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu9 = torch.nn.LeakyReLU()
        self.dropout9 = torch.nn.Dropout(p)

        self.linear10 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu10 = torch.nn.LeakyReLU()
        self.dropout10 = torch.nn.Dropout(p)

        self.linear11 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu11 = torch.nn.LeakyReLU()
        self.dropout11 = torch.nn.Dropout(p)

        self.linear12 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu12 = torch.nn.LeakyReLU()
        self.dropout12 = torch.nn.Dropout(p)

        self.linear13 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu13 = torch.nn.LeakyReLU()
        self.dropout13 = torch.nn.Dropout(p)

        self.linear14 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu14 = torch.nn.LeakyReLU()
        self.dropout14 = torch.nn.Dropout(p)

        self.linear15 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu15 = torch.nn.LeakyReLU()
        self.dropout15 = torch.nn.Dropout(p)

        self.linear_out = torch.nn.Linear(dim_hidden // 2, 1)

    def forward(self, vX):
        lin1 = self.linear1(vX)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        lin5 = self.linear5(drop4)
        h_relu5 = self.relu5(lin5)
        drop5 = self.dropout5(h_relu5)

        lin6 = self.linear6(drop5)
        h_relu6 = self.relu6(lin6)
        drop6 = self.dropout6(h_relu6)

        lin7 = self.linear7(drop6)
        h_relu7 = self.relu7(lin7)
        drop7 = self.dropout7(h_relu7)

        lin8 = self.linear8(drop7)
        h_relu8 = self.relu8(lin8)
        drop8 = self.dropout8(h_relu8)

        lin9 = self.linear9(drop8)
        h_relu9 = self.relu9(lin9)
        drop9 = self.dropout9(h_relu9)

        lin10 = self.linear10(drop9)
        h_relu10 = self.relu10(lin10)
        drop10 = self.dropout10(h_relu10)

        lin11 = self.linear11(drop10)
        h_relu11 = self.relu11(lin11)
        drop11 = self.dropout11(h_relu11)

        lin12 = self.linear12(drop11)
        h_relu12 = self.relu12(lin12)
        drop12 = self.dropout12(h_relu12)

        lin13 = self.linear13(drop12)
        h_relu13 = self.relu13(lin13)
        drop13 = self.dropout13(h_relu13)

        lin14 = self.linear14(drop13)
        h_relu14 = self.relu14(lin14)
        drop14 = self.dropout14(h_relu14)

        lin15 = self.linear15(drop14)
        h_relu15 = self.relu15(lin15)
        drop15 = self.dropout15(h_relu15)

        out = self.linear_out(drop15)
        return out

    def get_features(self, oa_origin, oa_destination, oa2features, oa2centroid, df, distances,k):
        return get_features_ffnn(oa_origin, oa_destination, oa2features, oa2centroid, df, distances,k)
