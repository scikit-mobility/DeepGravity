
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

from importlib.machinery import SourceFileLoader

path = './models/od_models.py'
od = SourceFileLoader('od', path).load_module()


def df_to_dict(df):
    split = df.to_dict(orient='split')
    keys = split['index']
    values = split['data']
    return {k: v for k, v in zip(keys, values)}

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

    n = len(oas)
    dim_train = int(n * fraction_train)

    random.shuffle(oas)
    train_locs = oas[:dim_train]
    test_locs = oas[dim_train:]

    return train_locs, test_locs


class NN_MultinomialRegression(od.NN_OriginalGravity):

    def __init__(self, dim_input, dim_hidden, df, dropout_p=0.35,  device=torch.device("cpu")):

        super(od.NN_OriginalGravity, self).__init__(dim_input, device=device)

        self.df = df

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

    def get_features(self, oa_origin, oa_destination, oa2features, oa2centroid, df):
        return get_features_ffnn(oa_origin, oa_destination, oa2features, oa2centroid, df)
