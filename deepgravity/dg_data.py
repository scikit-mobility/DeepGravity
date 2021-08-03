import torch
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import numpy as np

try:
    import od_models as od
except ModuleNotFoundError:
    import deepgravity.od_models as od


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    #target = torch.LongTensor(target)
    return [data, target], ids


class FlowDataset(torch.utils.data.Dataset):
    """
    partition
        {'train': ['tile_1', 'tile_2', 'tile_3'], 'test': ['tile_4']}
    """

    def __init__(self,
                 list_IDs: List[str],
                 tileid2oa2features2vals: Dict,
                 o2d2flow: Dict,
                 oa2features: Dict,
                 oa2pop: Dict,
                 oa2centroid: Dict,
                 dim_dests: int,
                 frac_true_dest: float, 
                 model: str
                ) -> None:
        'Initialization'
        self.list_IDs = list_IDs
        self.tileid2oa2features2vals = tileid2oa2features2vals
        self.o2d2flow = o2d2flow
        self.oa2features = oa2features
        self.oa2pop = oa2pop
        self.oa2centroid = oa2centroid
        self.dim_dests = dim_dests
        self.frac_true_dest = frac_true_dest
        self.model = model
        self.oa2tile = {oa:tile for tile,oa2v in tileid2oa2features2vals.items() for oa in oa2v.keys()}

    def __len__(self) -> int:
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get_features(self, oa_origin, oa_destination, df='exponential'):
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        oa2pop = self.oa2pop
        dist_od = od.earth_distance(oa2centroid[oa_origin], oa2centroid[oa_destination])
    
        if self.model in ['G']:
            if df == 'powerlaw':
                return [np.log(oa2features[oa_destination])] + [np.log(dist_od)]
            else:
                return [np.log(oa2features[oa_destination])] + [dist_od]
        elif self.model in ['NG']:
            if df == 'powerlaw':
                return [np.log(oa2features[oa_origin])] + [np.log(oa2features[oa_destination])] + [np.log(dist_od)]
            else:
                return [np.log(oa2features[oa_origin])] + [np.log(oa2features[oa_destination])] + [dist_od]
        else:
            return oa2features[oa_origin] + oa2features[oa_destination] + [dist_od]
            #return [np.log(oa2pop[oa_origin])] + oa2features[oa_origin] + \
            #       [np.log(oa2pop[oa_destination])] +  oa2features[oa_destination] + [dist_od]


    def get_flow(self, oa_origin, oa_destination):
        o2d2flow = self.o2d2flow
        try:
            # return od2flow[(oa_origin, oa_destination)]
            return o2d2flow[oa_origin][oa_destination]
        except KeyError:
            return 0

    def get_destinations(self, oa, size_train_dest, all_locs_in_train_region):
        o2d2flow = self.o2d2flow
        frac_true_dest = self.frac_true_dest
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

    def get_X_T(self, origin_locs, dest_locs):
        """
        origin_locs  :  list 1 X n_orig, IDs of origin locations
        dest_locs  :  list n_orig X n_dest, for each origin, IDs of destination locations
        """
        o2d2flow = self.o2d2flow
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid

        X, T = [], []
        for en, i in enumerate(origin_locs):
            X += [[]]
            T += [[]]
            for j in dest_locs[en]:
                X[-1] += [self.get_features(i, j)]
                T[-1] += [self.get_flow(i, j)]

        teX = torch.from_numpy(np.array(X)).float()
        teT = torch.from_numpy(np.array(T)).float()
        return teX, teT

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        'Generates one sample of data (one location)'

        tileid2oa2features2vals = self.tileid2oa2features2vals
        o2d2flow = self.o2d2flow
        oa2features = self.oa2features
        oa2centroid = self.oa2centroid
        dim_dests = self.dim_dests
        frac_true_dest = self.frac_true_dest
        oa2tile = self.oa2tile

        # Select sample (tile)
        sampled_origins = [self.list_IDs[index]]
        tile_ID = oa2tile[sampled_origins[0]]
        #print('tile_ID: %s'%tile_ID)

        # Load data and get flows

        # Select a subset of OD pairs
        all_locs_in_train_region = list(tileid2oa2features2vals[tile_ID].keys())
        size_train_dest = min(dim_dests, len(all_locs_in_train_region))
        sampled_dests = [self.get_destinations(oa, size_train_dest, all_locs_in_train_region)
                         for oa in sampled_origins]

        # get the features and flows
        sampled_trX, sampled_trT = self.get_X_T(sampled_origins, sampled_dests)

        return sampled_trX, sampled_trT, sampled_origins

