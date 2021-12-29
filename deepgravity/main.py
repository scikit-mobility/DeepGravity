from __future__ import print_function

import argparse

import torch.optim as optim
import torch.utils.data.distributed

import pandas as pd
import numpy as np

import random

import os

import time

from importlib.machinery import SourceFileLoader

# Training settings
parser = argparse.ArgumentParser(description='DeepGravity')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                    help='learning rate (default: 5e-6)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--mode', default='train', help='Can be train or test')
# Model arguments
parser.add_argument('--tessellation-area', default='United Kingdom',
                    help='The area to tessel if a tessellation is not provided')
parser.add_argument('--tessellation-size', type=int, default=25000,
                    help='The tessellation size (meters) if a tessellation is not provided')
parser.add_argument('--dataset', default='new_york', help='The dataset to use')

# Dataset arguments 
parser.add_argument('--tile-id-column', default='tile_ID', help='Column name of tile\'s identifier')
parser.add_argument('--tile-geometry', default='geometry', help='Column name of tile\'s geometry')

parser.add_argument('--oa-id-column', default='oa_ID', help='Column name of oa\'s identifier')
parser.add_argument('--oa-geometry', default='geometry', help='Column name of oa\'s geometry')

parser.add_argument('--flow-origin-column', default='origin', help='Column name of flows\' origin')
parser.add_argument('--flow-destination-column', default='destination', help='Column name of flows\' destination')
parser.add_argument('--flow-flows-column', default='flow', help='Column name of flows\' actual value')

args = parser.parse_args()

# global settings
model_type = 'DG'
data_name = args.dataset

# random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# loading DataLoader and utilities
path = './data_loader.py'
dgd = SourceFileLoader('dg_data', path).load_module()
path = './utils.py'
utils = SourceFileLoader('utils', path).load_module()

# set the device 
args.cuda = args.device.find("gpu") != -1

if args.device.find("gpu") != -1:
    torch.cuda.manual_seed(args.seed)
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")

# check if raw data exists and otherwise stop the execution
if not os.path.isdir('./data/' + data_name):
    raise ValueError('There is no dataset named ' + data_name + ' in ./data/')

db_dir = './data/' + data_name


def train(epoch):
    model.train()
    running_loss = 0.0
    training_acc = 0.0

    for batch_idx, data_temp in enumerate(train_loader):
        b_data = data_temp[0]
        b_target = data_temp[1]
        ids = data_temp[2]
        optimizer.zero_grad()
        loss = 0.0
        for data, target in zip(b_data, b_target):

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            output = model.forward(data)
            loss += model.loss(output, target)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            if batch_idx * len(b_data) == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} \tLoss: {:.6f}'.format(epoch, batch_idx * len(b_data), len(train_loader),
                                                                     loss.item() / args.batch_size))

    running_loss = running_loss / len(train_dataset)
    training_acc = training_acc / len(train_dataset)


def test():
    model.eval()
    with torch.no_grad():
        test_loss = 0.
        test_accuracy = 0.
        n_origins = 0
        for batch_idx, data_temp in enumerate(test_loader):
            b_data = data_temp[0]
            b_target = data_temp[1]
            ids = data_temp[2]
            test_loss = 0.0

            for data, target in zip(b_data, b_target):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                output = model.forward(data)
                test_loss += model.loss(output, target).item()

                cpc = model.get_cpc(data, target)
                test_accuracy += cpc
                n_origins += 1

            break

        test_loss /= n_origins
        test_accuracy /= n_origins


def evaluate():
    loc2cpc_numerator = {}

    model.eval()
    with torch.no_grad():
        for data_temp in test_loader:
            b_data = data_temp[0]
            b_target = data_temp[1]
            ids = data_temp[2]
            for id, data, target in zip(ids, b_data, b_target):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model.forward(data)
                cpc = model.get_cpc(data, target, numerator_only=True)
                loc2cpc_numerator[id[0]] = cpc
    edf = pd.DataFrame.from_dict(loc2cpc_numerator, columns=['cpc_num'], orient='index').reset_index().rename(
        columns={'index': 'locID'})
    oa2tile = {oa: t for t, v in tileid2oa2features2vals.items() for oa in v.keys()}

    def cpc_from_num(edf, oa2tile, o2d2flow):
        print(edf.head())
        edf['tile'] = edf['locID'].apply(lambda x: oa2tile[x])
        edf['tot_flow'] = edf['locID'].apply(lambda x: sum(o2d2flow[x].values()) if x in o2d2flow else 1e-6)
        cpc_df = pd.DataFrame(edf.groupby('tile').apply( \
            lambda x: x['cpc_num'].sum() / 2 / x['tot_flow'].sum()), \
            columns=['cpc']).reset_index()
        return cpc_df

    cpc_df = cpc_from_num(edf, oa2tile, o2d2flow)
    print('Average CPC of test tiles: {cpc_df.cpc.mean():.4f}  stdev: {cpc_df.cpc.std():.4f}')

    fname = './results/tile2cpc_{}_{}.csv'.format(model_type, args.dataset)

    cpc_df.to_csv(fname, index=False)


utils.tessellation_definition(db_dir, args.tessellation_area, args.tessellation_size)

tileid2oa2features2vals, oa_gdf, flow_df, oa2pop, oa2features, od2flow, oa2centroid = utils.load_data(db_dir,
                                                                                                      args.tile_id_column,
                                                                                                      args.tile_geometry,
                                                                                                      args.oa_id_column,
                                                                                                      args.oa_geometry,
                                                                                                      args.flow_origin_column,
                                                                                                      args.flow_destination_column,
                                                                                                      args.flow_flows_column)

oa2features = {oa: [np.log(oa2pop[oa])] + feats for oa, feats in oa2features.items()}

o2d2flow = {}
for (o, d), f in od2flow.items():
    try:
        d2f = o2d2flow[o]
        d2f[d] = f
    except KeyError:
        o2d2flow[o] = {d: f}

train_dataset_args = {'tileid2oa2features2vals': tileid2oa2features2vals,
                      'o2d2flow': o2d2flow,
                      'oa2features': oa2features,
                      'oa2pop': oa2pop,
                      'oa2centroid': oa2centroid,
                      'dim_dests': 512,
                      'frac_true_dest': 0.0,
                      'model': model_type}

test_dataset_args = {'tileid2oa2features2vals': tileid2oa2features2vals,
                     'o2d2flow': o2d2flow,
                     'oa2features': oa2features,
                     'oa2pop': oa2pop,
                     'oa2centroid': oa2centroid,
                     'dim_dests': int(1e9),
                     'frac_true_dest': 0.0,
                     'model': model_type}

# datasets
train_data = [oa for t in pd.read_csv(db_dir + '/processed/train_tiles.csv', header=None, dtype=object)[0].values for oa
              in tileid2oa2features2vals[str(t)].keys()]
test_data = [oa for t in pd.read_csv(db_dir + '/processed/test_tiles.csv', header=None)[0].values for oa in
             tileid2oa2features2vals[str(t)].keys()]

train_dataset = dgd.FlowDataset(train_data, **train_dataset_args)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

test_dataset = dgd.FlowDataset(test_data, **test_dataset_args)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size)

dim_input = len(train_dataset.get_features(train_data[0], train_data[0]))

if args.mode == 'train':

    model = utils.instantiate_model(oa2centroid, oa2features, oa2pop, dim_input, device=torch_device)
    if args.device.find("gpu") != -1:
        model.cuda()

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

    t0 = time.time()
    test()
    for epoch in range(1, args.epochs + 1):
        # set new random seeds
        torch.manual_seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        random.seed(args.seed + epoch)

        train(epoch)
        test()

    t1 = time.time()
    print("Total training time: %s seconds" % (t1 - t0))

    fname = './results/model_{}_{}.pt'.format(model_type, args.dataset)
    print('Saving model to {} ...'.format(fname))
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, fname)

    print('Computing the CPC on test set, loc2cpc_numerator ...')

    evaluate()


else:

    model = utils.instantiate_model(oa2centroid, oa2features, oa2pop, dim_input, device=torch_device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum)

    checkpoint = torch.load('./results/model_' + model_type + '_' + args.dataset + '.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    evaluate()
