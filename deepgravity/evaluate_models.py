import sys
import numpy as np
import pandas as pd
# import geopandas as gpd

# import deepgravity.od_models as od
# import deepgravity.ffnn as ffnn
import od_models as od
import ffnn as ffnn


def update_loc_dicts_with_new_tile(tile_id, oa2features, oa2centroid, od2flow, \
                                   tileid2oa2features2vals, oa_gdf, flow_df):
    oa2centroid_tile, oa2features_tile, od2flow_tile, \
    tileid2oa2features2vals, oa_gdf, flow_df = \
        ffnn.load_oa_features(tile_id=tile_id, \
                              tileid2oa2features=tileid2oa2features2vals, \
                              oa_gdf=oa_gdf, flow_df=flow_df)

    oa2features = {**oa2features, **oa2features_tile}
    oa2centroid = {**oa2centroid, **oa2centroid_tile}
    od2flow = {**od2flow, **od2flow_tile}

    return oa2features, oa2centroid, od2flow, tileid2oa2features2vals, oa_gdf, flow_df


def create_o2d2flow(od2flow):
    o2d2flow = {}
    for (o, d), f in od2flow.items():
        try:
            d2f = o2d2flow[o]
            d2f[d] = f
        except KeyError:
            o2d2flow[o] = {d: f}
    return o2d2flow


def match_flows_model_observ(flatten_model, flatten_observed, od_pairs, oa2cgt):
    """
    # LSOA
    oa2cgt = oa2lsoa
    # MSOA
    oa2cgt = oa2msoa

    """
    aggr_flows = {}
    oa_fl = {}

    for en, (o, d) in enumerate(od_pairs):

        aggr_flow_model = flatten_model[en]
        aggr_flow_obs = flatten_observed[en]
        oa_fl[(o, d)] = [aggr_flow_model, aggr_flow_obs]

        try:
            lo = oa2cgt[o]
            ld = oa2cgt[d]
        except KeyError:
            # print(o, d)
            continue

        if aggr_flow_model > 0 or aggr_flow_obs > 0:
            try:
                aggr_flows[(lo, ld)][0] += aggr_flow_model
                aggr_flows[(lo, ld)][1] += aggr_flow_obs
            except KeyError:
                aggr_flows[(lo, ld)] = [aggr_flow_model, aggr_flow_obs]


    cg_flow_model, cg_flow_observ = list(zip(*aggr_flows.values()))
    oa_flow_model, oa_flow_observ = list(zip(*oa_fl.values()))

    return cg_flow_model, cg_flow_observ, oa_flow_model, oa_flow_observ


def split_into_chuncks_of_max_size(max_size, l):
    return np.split(l, np.arange(0, len(l), max_size, dtype=int)[1:])


# def evaluate_model_performance(all_test_tiles, model, oa2cg, \
#                                oa2features, oa2centroid, od2flow, tileid2oa2features2vals, oa_gdf, flow_df, \
#                                max_od=100000, min_oa_in_tile=20, verbose=True, return_flows=False):
def evaluate_model_performance(all_test_tiles, model, oa2cg, \
                               tileid2oa2features2vals, oa_gdf, flow_df, \
                               max_od=100000, min_oa_in_tile=20, verbose=True, return_flows=False):
    """
    # all_test_tiles = ['680']  # '382', '680', '278'

    oa2cg = oa2msoa
    """
    tile2performance = {}
    tot_tiles = len(all_test_tiles)

    if 'OriginalGravity' in str(type(model)):
        # Resident population
        oa_pop = flow_df.groupby('residence').sum()
        oa2pop = oa_pop.to_dict()['commuters']
        # add oa's with 0 population
        all_oas = set(oa_gdf['geo_code'].values)
        oa2pop = {**{o: 1e-6 for o in all_oas}, **oa2pop}

    for ii, tile_id in enumerate(all_test_tiles):

        #     sys.stdout.write(" tile %s: %s of %s ...                                          \r"%(tile_id, ii + 1, tot_tiles))
        all_locs_in_test_region = list(tileid2oa2features2vals[tile_id].keys())
        if len(all_locs_in_test_region) < min_oa_in_tile:
            continue

        # add locations in tile to the dictionaries
        oa2features, oa2centroid, od2flow, tileid2oa2features2vals, oa_gdf, flow_df = \
            update_loc_dicts_with_new_tile(tile_id, {}, {}, {},
                                           tileid2oa2features2vals, oa_gdf, flow_df)
        # update_loc_dicts_with_new_tile(tile_id, oa2features, oa2centroid, od2flow,
        #                                tileid2oa2features2vals, oa_gdf, flow_df)
        o2d2flow = create_o2d2flow(od2flow)

        if sum(list(od2flow.values())) == 0:
            continue


        # compute the chunk size of origin locations
        # so that the product (number origin) * (number destination) is less than `max_od`
        max_size = max(1, max_od // len(all_locs_in_test_region))

        flatten_model = []
        flatten_observed = []
        od_pairs = []
        nlogl = 0.0

        chunks = split_into_chuncks_of_max_size(max_size, all_locs_in_test_region)
        for jj, test_locs in enumerate(chunks):

            if verbose:
                sys.stdout.write(" tile %s: %s of %s. chunk %s of %s...                                          \r" % \
                                 (tile_id, ii + 1, tot_tiles, jj + 1, len(chunks)))

            test_dests = [all_locs_in_test_region for _ in test_locs]

            if 'OriginalGravity' in str(type(model)):
                teX, teT = model.get_X_T(test_locs, test_dests, oa2pop, oa2centroid, o2d2flow,
                                         verbose=False)
            else:
                teX, teT = model.get_X_T(test_locs, test_dests, oa2features, oa2centroid, o2d2flow,
                                         verbose=False)

            model_OD = model.average_OD_model(teX, teT)

            od_pairs += [[o, d] for i, o in enumerate(test_locs) for d in test_dests[i]]
            flatten_model += list(model_OD.flatten())
            if 'cuda' in model.device.type:
                flatten_observed += list(teT.cpu().detach().numpy().flatten())
            else:
                flatten_observed += list(teT.detach().numpy().flatten())


            # negative loglikelihhod
            nlogl += model.negative_loglikelihood(teX, teT)


        cg_flow_model, cg_flow_observ, oa_flow_model, oa_flow_observ = \
            match_flows_model_observ(flatten_model, flatten_observed, od_pairs, oa2cg)

        # cpc and correlation
        cpc_oa = od.common_part_of_commuters(oa_flow_model, oa_flow_observ)
        corr_oa = np.corrcoef(oa_flow_model, oa_flow_observ)[0, 1]
        cpc = od.common_part_of_commuters(cg_flow_model, cg_flow_observ)
        corr = np.corrcoef(cg_flow_model, cg_flow_observ)[0, 1]

        # save all performance metrics of the tile
        tile2performance[tile_id] = {}
        perf2val = tile2performance[tile_id]
        perf2val['nlogl'] = nlogl
        perf2val['corr_oa'] = corr_oa
        perf2val['corr_msoa'] = corr
        perf2val['cpc_oa'] = cpc_oa
        perf2val['cpc_msoa'] = cpc

    if return_flows:
        return tile2performance, od_pairs, cg_flow_model, cg_flow_observ, oa_flow_model, oa_flow_observ
    else:
        return tile2performance
