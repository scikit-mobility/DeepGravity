import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from math import sqrt, sin, cos, pi, asin

#import deepgravity


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


def common_part_of_commuters(values1, values2, numerator_only=False):
    """
    Compute the common part of commuters for two pairs of fluxes.

    :param values1: the values for the first array
    :type values1: numpy array

    :param values2: the values for the second array
    :type values1: numpy array

    :return: float
        the common part of commuters
    """
    if numerator_only:
        tot = 1.0
    else:
        tot = (np.sum(values2) + np.sum(values2))
    if tot > 0:
        return 2.0 * np.sum(np.minimum(values1, values2)) / tot
    else:
        return 0.0

#   Generate synthetic flows to test algorithms

def Eucl_Dist(A, B):
    """
    Euclidean distance between points A=(x1,y1) and B=(x2,y2).
    """
    return np.hypot( A[0]-B[0], A[1]-B[1])


def generate_synthetic_flows_scGM(n_locations, weights, coordinates,\
        tot_outflows, deterrence_function, distfunc=Eucl_Dist):
    """
    Generate synthetic data accoding to the assumptions:
       1. Assume $n$ locations are randomly and uniformy distributed
       2. generate a random array of $w_i = T_i$
       3. generate the distance matrix $r_{ij}$
       4. generate the matrix $p_{ij}$
       5. generate a sample of $T_{ij}$ according to the multinomial distribution

    Returns:
        r : matrix of distances between all pairs of locations
        p : matrix of probabilities of one trip between all pairs of locations
        x : matrix of flows (one realization of the process)


    Example:

        n = 30
        n_locations = n
        R_true = 50.
        L = 500.0

        weights = 1.*np.random.randint(100,10000, size=n)
        coordinates = L*np.random.random(size=(n,2))
        tot_outflows = (1.-np.random.random(n)*0.5)*weights   #200*np.ones(n)

        # exponential deterrence function
        deterrence_function = (lambda x: np.exp(-x**1./R_true) )
        # power law deterrence function
        # deterrence_function = (lambda x: np.power(x,3.) )

        rr, probs, flows = generate_synthetic_flows_scGM(n_locations, weights, coordinates,\
                tot_outflows, deterrence_function, distfunc=Eucl_Dist)

        np.round(p[:5,:5],2)
        np.round(x[:5,:5],2)

    """
    n = n_locations
    w = weights
    ll= coordinates

    r = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            r[i,j] = distfunc(ll[i],ll[j])

    p = deterrence_function(r)
    p -= np.eye(n)*np.diag(p)

    tpw = np.transpose(p*w)
    p = np.transpose(tpw/sum(tpw))
    np.putmask(p,np.isnan(p),0.0)

    x = np.array([np.random.multinomial(tot_outflows[i], p[i]) for i in range(n)])

    return r, p, x


#  GLM Multinomial regression with mini-batches
class GLM_MultinomialRegression(torch.nn.Module):
    """

    to run on CPU
        device = torch.device("cpu")

    to run on GPU
        device = torch.device("cuda:0") # Uncomment this to run on GPU


    # Define variables
    dim_train = 10
    dim_test  = 10

    ii = np.random.choice(range(n), size=dim_train+dim_test, replace=False)

    jj = np.random.randint(5,len(ii), size=len(ii))
    X  = [np.array([[np.log(weights[j]), -r[i,j]] for j in ii[:jj[i0]] if j!=i], dtype=float) for i0,i in enumerate(ii)]
    T  = [np.array([1.*x[i,j] for j in ii[:jj[i0]] if j!=i], dtype=float) for i0,i in enumerate(ii)]

    trX = X[:dim_train]
    trT = T[:dim_train]

    teX = X[dim_train:]
    teT = T[dim_train:]

    print(len(trX),len(trX[0]),len(trX[0][0]))
    print(len(trT),len(trT[0]))

    # X  = np.array([[[np.log(weights[j]), -r[i,j]] for j in ii if j!=i] for i in ii], dtype=float)
    # T  = np.array([[x[i,j] for j in ii if j!=i] for i in ii], dtype=float)

    # trX = torch.from_numpy(X[:dim_train,:dim_train]).float()
    # trT = torch.from_numpy(T[:dim_train,:dim_train]).float()

    # teX = torch.from_numpy(X[dim_train:,dim_train:]).float()
    # teT = torch.from_numpy(T[dim_train:,dim_train:]).float()

    # print(trX.size())
    # print(trT.size())


    #dim_w = trX.size()[2]
    dim_w = len(trX[0][0])
    model = GLM_MultinomialRegression(dim_w)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e2)

    for i in range(500):
        cost = model.train_one(optimizer, trX, trT)
    #     a1,a2 = list(model.parameters())[0].data.numpy()[0]
        if i%10==0:
            print(i, np.round(cost,3)) #, a1, 1./a2)
    # a1,a2 = list(model.parameters())[0].data.numpy()[0]
    # print(a1, 1./a2)

    """
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
        """
        tX and tY are lists of numpy arrays (that can have variable dimensions)

        """
        # Reset gradient
        optimizer.zero_grad()

#        x = Variable(torch.FloatTensor(tX[0]), requires_grad=False)
#        y = Variable(torch.FloatTensor(tY[0]), requires_grad=False)
#
#        # Forward
#        fx = self.forward(x)
#        NlogL = self.loss(fx, y)

        NlogL = 0.

        num_batches = len(tX)
        for k in range(num_batches):
            # x = Variable(torch.FloatTensor(tX[k]), requires_grad=False)
            # y = Variable(torch.FloatTensor(tY[k]), requires_grad=False)
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

    # def validation(self, teX, teT):
    #     """
    #     teX and teT are numpy arrays
    #
    #     xx,yy = model.validation(teX,teT)
    #
    #     fig, ax = plt.subplots()
    #
    #     plt.plot(xx, yy, 'o', label='model')
    #
    #     xx = np.linspace(min(xx),max(xx),10)
    #     plt.plot(xx,xx)
    #
    #     # plt.legend(loc='best')
    #     ax.set_title('Model Fit Plot')
    #     ax.set_ylabel('true')
    #     ax.set_xlabel('model')
    #
    #     # plt.xscale('log')
    #     # plt.yscale('log')
    #     plt.show()
    #
    #     """
    #     model_p = np.array([self.predict_proba_pth(xx) for xx in teX], \
    #                         dtype=float).astype('float64')
    #     totT = np.sum(teT, axis=1)
    #     predicted_T = np.array([np.random.multinomial(totT[i], pp/sum(pp)) \
    #                  for i,pp in enumerate(model_p)])
    #
    #     xx = predicted_T.flatten()
    #     yy = teT.flatten()
    #     return xx, yy


def get_features_original_gravity(oa_origin, oa_destination, oa2features, oa2centroid, df='exponential'):
    dist_od = earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
    #if df == 'powerlaw':
    #    return [np.log(oa2features[oa_destination])] + [np.log(dist_od)]
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
        """
        dim_input = 2
        dim_hidden = 20
        """
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
        """
        origin_locs  :  list 1 X n_orig, IDs of origin locations
        dest_locs  :  list n_orig X n_dest, for each origin, IDs of destination locations
        """

        #print(origin_locs)s

        # n_locs = len(origin_locs)
        X, T = [], []
        for en, i in enumerate(origin_locs):
            if verbose:
                pass
                # clear_output(wait=True)
                # print('%s of %s...'%(en, n_locs))
            #         o_lalo = oa2centroid[i]
            #         o_feats = oa2features[i]
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                if self.df == 'exponential' or self.df == 'exponential_all':
                    X[-1] += [self.get_features(i, j, oa2features, oa2centroid, self.df)]
                else:
                    X[-1] += [self.get_features(i, j, oa2features, oa2centroid, self.df, self.distances, self.k)]
                #X[-1] += [self.get_features(i, j, oa2features, oa2centroid, 'deepgravity_people')]
                #             X[-1] += [get_features(o_lalo, o_feats, j, oa2features, oa2centroid)]
                T[-1] += [get_flow(i, j, o2d2flow)]

        # teX = torch.FloatTensor(X, device=self.device)
        # teT = torch.FloatTensor(T, device=self.device)
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

    def train_many_steps(self, steps, train_locs, test_locs,
                         all_locs_in_train_region, all_locs_in_test_region,
                         oa2features, oa2centroid, o2d2flow,
                         optimizer, dim_origins=50, dim_dests=500, save_every=20, verbose=True):

        # oa_within = list(oa2centroid.keys())

        # test_dests = [oa_within for _ in test_locs]
        size = min(dim_dests, len(all_locs_in_test_region))
        test_dests = [np.random.choice(all_locs_in_test_region, size=size, replace=False)
                      for _ in test_locs]
        teX, teT = self.get_X_T(test_locs, test_dests, oa2features, oa2centroid, o2d2flow, verbose=False)

        size_train = min(dim_dests, len(all_locs_in_train_region))

        train_losses = []
        test_losses = []
        cpc_trains = []
        cpc_tests = []

        if verbose:
            print('epoch loss_train loss_test')

        for i in range(steps):

            # Select a subset of OD pairs

            #     # randomly select a subset of origins
            #     sampled_origins = np.random.choice(range(trX.shape[0]), size=dim_origins, replace=False)
            #     trX_sO = trX[sampled_origins, :, :]
            #     trT_sO = trT[sampled_origins, :]

            #     # OR select all origins
            # #     trX_sO = trX
            # #     trT_sO = trT

            #     # randomly select a subset of `dim_dests` destinations from train_locs

            #     # SAME destinations for all origins
            # #     sampled_dests = np.random.choice(range(trX_sO.shape[1]), size=dim_dests, replace=False)
            # #     sampled_trX = trX_sO[:, sampled_dests, :]
            # #     sampled_trT = trT_sO[:, sampled_dests]

            #     # DIFFERENT destinations for each origin
            #     d0, d1, d2 = trX_sO.shape
            #     sampled_dests = [np.random.choice(range(d1), size=dim_dests, replace=False) for s in range(d0)]
            #     sampled_trX = torch.cat([trX_sO[s, locs, :] for s, locs in enumerate(sampled_dests)]).view(d0, dim_dests, d2)
            #     sampled_trT = torch.cat([trT_sO[s, locs] for s, locs in enumerate(sampled_dests)]).view(d0, dim_dests)

            #     # OR consider ALL destinations
            # #     sampled_trX = trX_sO
            # #     sampled_trT = trT_sO

            #     # create features on the fly

            #     sampled_origins = np.random.choice(train_locs, size=dim_origins, replace=False)
            #     sampled_dests = [np.random.choice(ii, size=dim_dests, replace=False) for s in range(dim_origins)]

            #     X  = np.array([[get_features(oa_within[i], oa_within[j], oa_features) for j in sampled_dests[en]] # if j!=i]
            #                    for en, i in enumerate(sampled_origins)], dtype=float)
            #     T  = np.array([[get_flow(oa_within[i], oa_within[j], flow_df_within) for j in sampled_dests[en]] # if j!=i]
            #                    for en, i in enumerate(sampled_origins)], dtype=float)
            #     sampled_trX = torch.from_numpy(X).float()
            #     sampled_trT = torch.from_numpy(T).float()

            sampled_origins = np.random.choice(train_locs, size=dim_origins, replace=False)
            sampled_dests = [np.random.choice(all_locs_in_train_region, size=size_train, replace=False)
                             for _ in sampled_origins]
            sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests, oa2features, oa2centroid, o2d2flow)

            # Train

            loss_train = self.train_one(optimizer, sampled_trX, sampled_trT)

            if i % save_every == 0:
                self.eval()

                train_losses += [loss_train]
                loss_test = self.loss(self.forward(teX), teT).item()
                test_losses += [loss_test]

                # flatten_test_observed = teT.detach().numpy().flatten()
                # model_OD_test = self.average_OD_model(teX, teT)
                # flatten_test_model = model_OD_test.flatten()
                # cpc_test = common_part_of_commuters(flatten_test_observed, flatten_test_model)
                cpc_test = self.get_cpc(teX, teT)
                cpc_tests += [cpc_test]

                # flatten_train_observed = sampled_trT.detach().numpy().flatten()
                # model_OD_train = self.average_OD_model(sampled_trX, sampled_trT)
                # flatten_train_model = model_OD_train.flatten()
                # cpc_train = common_part_of_commuters(flatten_train_observed, flatten_train_model)
                cpc_train = self.get_cpc(sampled_trX, sampled_trT)
                cpc_trains += [cpc_train]

                self.train()

                if verbose:
                    print(i, np.round(loss_train, 2), np.round(loss_test, 2))

        return optimizer, train_losses, test_losses, cpc_trains, cpc_tests

    def train_many_steps_loop_multiple_tiles(self, steps, train_tiles, test_tile,
                         oa2features, oa2centroid, o2d2flow, tileid2oa2features2vals,
                         optimizer, dim_origins=4, dim_dests=500, save_every=20, frac_true_dest=0.5, verbose=True):

        # test_locs = list(tileid2oa2features2vals[test_tile].keys())
        # all_locs_in_test_region = test_locs
        all_locs_in_test_region = list(tileid2oa2features2vals[test_tile].keys())
        size = min(dim_dests, len(all_locs_in_test_region))

        test_locs = np.random.choice(all_locs_in_test_region, size=size, replace=False)

        # test_dests = [oa_within for _ in test_locs]
        size = min(dim_dests, len(all_locs_in_test_region))
        test_dests = [np.random.choice(all_locs_in_test_region, size=size, replace=False)
                      for _ in test_locs]
        teX, teT = self.get_X_T(test_locs, test_dests, oa2features, oa2centroid, o2d2flow, verbose=False)

        train_losses = []
        test_losses = []
        cpc_trains = []
        cpc_tests = []


        if verbose:
            print('epoch loss_train loss_test')

        for i in range(steps):

            # Reset gradient
            optimizer.zero_grad()
            NlogL = 0.

            # Loop over all training tiles and sample a fixed number of locations from each
            # Here `dim_origins` refers to the number of locations in each single tile
            for train_tile in train_tiles:
            # Select a tile
            # train_tile = np.random.choice(train_tiles)

                train_locs = list(tileid2oa2features2vals[train_tile].keys())
                all_locs_in_train_region = train_locs
                size_train_orig = min(dim_origins, len(all_locs_in_train_region))
                size_train_dest = min(dim_dests, len(all_locs_in_train_region))

                # Select a subset of OD pairs

                sampled_origins = np.random.choice(train_locs, size=size_train_orig, replace=False)
                # sampled_dests = [np.random.choice(all_locs_in_train_region, size=size_train_dest, replace=False)
                #                  for _ in sampled_origins]
                sampled_dests = [get_destinations(oa, size_train_dest, all_locs_in_train_region, o2d2flow,
                                                  frac_true_dest=frac_true_dest) for oa in sampled_origins]
                sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests, oa2features, oa2centroid, o2d2flow)



                # Forward
                fx = self.forward(sampled_trX)
                NlogL += self.loss(fx, sampled_trT)

            # Backward
            NlogL.backward()

            # Update parameters
            optimizer.step()

            # compute loss
            loss_train = NlogL.item()


            # Train
            # loss_train = self.train(optimizer, sampled_trX, sampled_trT)

            if i % save_every == 0:
                self.eval()

                train_losses += [loss_train]
                loss_test = self.loss(self.forward(teX), teT).item()
                test_losses += [loss_test]

                cpc_test = self.get_cpc(teX, teT)
                cpc_tests += [cpc_test]

                cpc_train = self.get_cpc(sampled_trX, sampled_trT)
                cpc_trains += [cpc_train]

                self.train()

                if verbose:
                    print(i, np.round(loss_train, 2), np.round(loss_test, 2))

        return optimizer, train_losses, test_losses, cpc_trains, cpc_tests

    def train_many_steps_multiple_tiles(self, steps, train_tiles, test_tile,
                         oa2features, oa2centroid, o2d2flow, tileid2oa2features2vals,
                         optimizer, dim_origins=150, dim_dests=500, save_every=20, frac_true_dest=0.5, verbose=True):
        # test_locs = list(tileid2oa2features2vals[test_tile].keys())
        # all_locs_in_test_region = test_locs
        all_locs_in_test_region = list(tileid2oa2features2vals[test_tile].keys())

        size = min(dim_dests, len(all_locs_in_test_region))
        test_locs = np.random.choice(all_locs_in_test_region, size=size, replace=False)

        size = min(dim_dests, len(all_locs_in_test_region))
        test_dests = [np.random.choice(all_locs_in_test_region, size=size, replace=False)
                      for _ in test_locs]
        teX, teT = self.get_X_T(test_locs, test_dests, oa2features, oa2centroid, o2d2flow, verbose=False)

        train_losses = []
        test_losses = []
        cpc_trains = []
        cpc_tests = []

        # train tiles and locs
        loc2tile_train = {l: t for t in train_tiles for l in list(tileid2oa2features2vals[t].keys())}
        train_locs_all_tiles = list(loc2tile_train.keys())

        if verbose:
            print('epoch loss_train loss_test')

        for i in range(steps):

            # Reset gradient
            optimizer.zero_grad()
            NlogL = 0.


            # Sample locations from all tiles
            # Here `dim_origins` refers to the number of all locations in all tiles
            sample_train_locs = np.random.choice(train_locs_all_tiles, size=dim_origins, replace=False)
            for train_tile, v in pd.DataFrame([[l, loc2tile_train[l]] for l in sample_train_locs]).groupby(1):
                sampled_origins = v[0].values

                train_locs = list(tileid2oa2features2vals[train_tile].keys())
                all_locs_in_train_region = train_locs
                size_train_dest = min(dim_dests, len(all_locs_in_train_region))

                # Select a subset of OD pairs

                # sampled_dests = [np.random.choice(all_locs_in_train_region, size=size_train_dest, replace=False)
                #                  for _ in sampled_origins]
                sampled_dests = [get_destinations(oa, size_train_dest, all_locs_in_train_region, o2d2flow,
                                                  frac_true_dest=frac_true_dest) for oa in sampled_origins]

                sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests, oa2features, oa2centroid,
                                                        o2d2flow)


                # Forward
                fx = self.forward(sampled_trX)
                NlogL += self.loss(fx, sampled_trT)


            # Backward
            NlogL.backward()

            # Update parameters
            optimizer.step()

            # compute loss
            loss_train = NlogL.item()


            # Train
            # loss_train = self.train(optimizer, sampled_trX, sampled_trT)

            if i % save_every == 0:
                self.eval()

                train_losses += [loss_train]
                loss_test = self.loss(self.forward(teX), teT).item()
                test_losses += [loss_test]

                cpc_test = self.get_cpc(teX, teT)
                cpc_tests += [cpc_test]

                cpc_train = self.get_cpc(sampled_trX, sampled_trT)
                cpc_trains += [cpc_train]

                self.train()

                if verbose:
                    print(i, np.round(loss_train, 2), np.round(loss_test, 2))

        return optimizer, train_losses, test_losses, cpc_trains, cpc_tests
