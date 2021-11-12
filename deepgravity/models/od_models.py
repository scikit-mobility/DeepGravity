import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from math import sqrt, sin, cos, pi, asin

#import deepgravity


def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...


def common_part_of_commuters(values1, values2, numerator_only=False):
    if numerator_only:
        tot = 1.0
    else:
        tot = (np.sum(values2) + np.sum(values2))
    if tot > 0:
        return 2.0 * np.sum(np.minimum(values1, values2)) / tot
    else:
        return 0.0

class GLM_MultinomialRegression(torch.nn.Module):
    def __init__(self, dim_w, device=torch.device("cpu")):
        super(GLM_MultinomialRegression, self).__init__()

        self.device = device
        self.linear1 = torch.nn.Linear(dim_w, 1)

    def forward(self, vX):
        out = self.linear1(vX)
        return out

    def loss(self, out, vT):
        lsm = torch.nn.LogSoftmax(dim=1)
        return -( vT * lsm(torch.squeeze(out, dim=-1)) ).sum()

    def negative_loglikelihood(self, tX, tT):
        return self.loss(self.forward(tX), tT).item()

    def train_one(self, optimizer, tX, tY):
        
        # Reset gradient
        optimizer.zero_grad()

        NlogL = 0.

        num_batches = len(tX)
        for k in range(num_batches):
            if 'cuda' in self.device.type:
                x = Variable(torch.from_numpy(np.array(tX[k])).cuda(), requires_grad=False)
                y = Variable(torch.from_numpy(np.array(tY[k])).cuda(), requires_grad=False)
            else:
                x = Variable(torch.from_numpy(np.array(tX[k])), requires_grad=False)
                y = Variable(torch.from_numpy(np.array(tY[k])), requires_grad=False)

            # Forward
            fx = self.forward(x)
            NlogL += self.loss(fx, y)

        # Backward
        NlogL.backward()

        # Update parameters
        optimizer.step()

        return NlogL.item()  #NlogL.data[0]

    def predict_proba(self, x):
        sm = torch.nn.Softmax(dim=1)
        #probs = sm(torch.squeeze(self.forward(x), dim=2))
        probs = sm(torch.squeeze(self.forward(x), dim=-1))
        if 'cuda' in self.device.type:
            return probs.cpu().detach().numpy()
        else:
            return probs.detach().numpy()

    def average_OD_model(self, tX, tT):
        p = self.predict_proba(tX)
        if 'cuda' in self.device.type:
            #tot_out_trips = tT.sum(dim=1).cpu().detach().numpy()
            tot_out_trips = tT.sum(dim=-1).cpu().detach().numpy()
        else:
            #tot_out_trips = tT.sum(dim=1).detach().numpy()
            tot_out_trips = tT.sum(dim=-1).detach().numpy()
        model_od = (p.T * tot_out_trips).T
        return model_od

    def predict(self, x_val):
        x = Variable(x_val, requires_grad=False)
        output = self.forward(x)
        #return output.data.numpy().argmax(axis=1)
        return output.data.argmax(axis=1)

def get_features_original_gravity(oa_origin, oa_destination, oa2features, oa2centroid, df='exponential'):
    dist_od = earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
    if df == 'exponential':
        return [np.log(oa2features[oa_destination])] + [dist_od]
    elif df == 'exponential_all':
        return oa2features[oa_origin] + oa2features[oa_destination] + [dist_od]

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

class NN_OriginalGravity(GLM_MultinomialRegression):

    def __init__(self, dim_input, df='exponential', device=torch.device("cpu")):

        super(GLM_MultinomialRegression, self).__init__()
        #         super().__init__(self)
        self.device = device
        self.df = df
        self.dim_input = dim_input
        self.linear_out = torch.nn.Linear(dim_input, 1)

    def forward(self, vX):
        out = self.linear_out(vX)
        return out

    def get_features(self, oa_origin, oa_destination, oa2features, oa2centroid, df):
        return get_features_original_gravity(oa_origin, oa_destination, oa2features, oa2centroid, df=df)

    def get_X_T(self, origin_locs, dest_locs, oa2features, oa2centroid, o2d2flow, verbose=False):
       
        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                X[-1] += [self.get_features(i, j, oa2features, oa2centroid, self.df)]
              
                T[-1] += [get_flow(i, j, o2d2flow)]

        if 'cuda' in self.device.type:
            teX = torch.from_numpy(np.array(X)).float().cuda()
            teT = torch.from_numpy(np.array(T)).float().cuda()
        else:
            teX = torch.from_numpy(np.array(X)).float()
            teT = torch.from_numpy(np.array(T)).float()

        return teX, teT

    def get_cpc(self, teX, teT, numerator_only=False):
        if 'cuda' in self.device.type:
            flatten_test_observed = teT.cpu().detach().numpy().flatten()
        else:
            flatten_test_observed = teT.detach().numpy().flatten()
        model_OD_test = self.average_OD_model(teX, teT)
        flatten_test_model = model_OD_test.flatten()
        cpc_test = common_part_of_commuters(flatten_test_observed, flatten_test_model, 
                                            numerator_only=numerator_only)
        return cpc_test