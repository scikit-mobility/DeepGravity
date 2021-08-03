from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import pandas as pd
import numpy as np
import random

# Horovod: import horovod 
import horovod.torch as hvd
from horovod.torch.mpi_ops import Average, Adasum, Sum
# Horovod: initialize library.
hvd.init()
print("Horovod: I am worker %s of %s." %(hvd.rank(), hvd.size()))

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import time

# TODO: add as arguments
tile_size = '25000'

# Training settings
parser = argparse.ArgumentParser(description='DeepGravity')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                    help='learning rate (default: 5e-6)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device', default='cpu',
                    help='Wheter this is running on cpu or gpu')
parser.add_argument('--num_threads', default=0, help='set number of threads per worker', type=int)

# model args
parser.add_argument('--model-type', default='DG',
                    help='G, NG, MFG, DG, DGknn')
parser.add_argument('--country', default='uk',
                    help='uk, ita')
parser.add_argument('--no-round', type=int, default=0,
                    help='different splits of train and test sets: 0, 1, 2')

args = parser.parse_args()

model_type = args.model_type
country = args.country
no_round = args.no_round

# random seeds
torch.manual_seed(args.seed + hvd.rank())
np.random.seed(args.seed + hvd.rank())
random.seed(args.seed + hvd.rank())

# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):                             
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(np.random.get_state()[1][0] + worker_id)

# data and model
from importlib.machinery import SourceFileLoader
path = './dg_data.py'
dgd = SourceFileLoader('dg_data', path).load_module()
path = './utils.py'
utils = SourceFileLoader('utils', path).load_module()


args.cuda = args.device.find("gpu")!=-1

if args.device.find("gpu")!=-1:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
    torch_device = torch.device("cuda")
else:
    torch_device = torch.device("cpu")

if (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

if hvd.rank()==0:
    print(args)
    print("Torch Thread setup: ")
    print(" Number of threads: ", torch.get_num_threads())
#    print(" Number of inter_op threads: ", torch.get_num_interop_threads())


#=====================================================================================

# Define dataset

# parameters
if args.device.find("gpu") != -1:
    kwargs = {#'shuffle': True,
              'collate_fn': dgd.my_collate,
              'worker_init_fn': worker_init_fn,
              'num_workers': 1,
              'pin_memory': True}
else:
    kwargs = {'collate_fn': dgd.my_collate, 
              'worker_init_fn': worker_init_fn}

wk_dir = '../'
data_dir = 'data/' + country + '/'
db_dir = wk_dir + data_dir
tileid2oa2features2vals, oa_gdf, flow_df, msoa_df, oa2msoa, oa2pop, \
        oa2features, od2flow, oa2centroid = utils.load_data(db_dir, tile_size=tile_size)

# update features
if model_type in ['G', 'NG']:
    oa2features = oa2pop
else: #elif model_type in ['DG', 'MFG']:
    oa2features = {oa: [np.log(oa2pop[oa])] + feats for oa,feats in oa2features.items()}

if model_type == 'DGknn':
    oa2closestk = utils.load_oa2closestk(db_dir, k=8)
    oa2features = utils.compute_oa2featuresk(oa2closestk, oa2features, k=4)

o2d2flow = {}
for (o, d),f in od2flow.items():
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
train_data = [oa for t in 
               pd.read_csv(db_dir + tile_size + '/train_tiles_' + str(no_round) + '.csv', 
                           header=None)[0].values \
              for oa in tileid2oa2features2vals[str(t)].keys()]

test_data = [oa for t in 
               pd.read_csv(db_dir + tile_size + '/test_tiles_' + str(no_round) + '.csv', 
                           header=None)[0].values \
              for oa in tileid2oa2features2vals[str(t)].keys()]


train_dataset = dgd.FlowDataset(train_data, **train_dataset_args)

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)


test_dataset = dgd.FlowDataset(test_data, **test_dataset_args)

# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

# Define model
dim_input = len(train_dataset.get_features(train_data[0], train_data[0]))
model = utils.instantiate_model(oa2centroid, oa2features, oa2pop, dim_input, device=torch_device, df=model_type)

if args.device.find("gpu") != -1:
    # Move model to GPU.
    model.cuda()

if hvd.rank() == 0:
    print(model.device.type)

#=====================================================================================


# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.RMSprop(model.parameters(), lr=args.lr * hvd.size(),
                      momentum=args.momentum)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

## Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
									 named_parameters=model.named_parameters(),
									 compression=compression,
                                     op=Sum)


def train(epoch):
    model.train()
    running_loss = 0.0
    training_acc = 0.0
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, ([b_data, b_target], ids) in enumerate(train_loader):

        optimizer.zero_grad()
        loss = 0.0 #torch.tensor(0.0, requires_grad=True)

        for data, target in zip(b_data, b_target):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #output = model(data)
            #loss = F.nll_loss(output, target)
            output = model.forward(data)
            loss += model.loss(output, target)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(hvd.rank(), 
                epoch, batch_idx * len(b_data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()/args.batch_size))

    running_loss = running_loss / len(train_sampler)
    training_acc = training_acc / len(train_sampler)
    loss_avg = metric_average(running_loss, 'running_loss')
    training_acc = metric_average(training_acc, 'training_acc')

    if hvd.rank()==0: 
        print("Training set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(loss_avg, training_acc*100))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    with torch.no_grad():
        test_loss = 0.
        test_accuracy = 0.
        n_origins = 0
        for [b_data, b_target], ids in test_loader:
    
            test_loss = 0.0
    
            for data, target in zip(b_data, b_target):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model.forward(data)
                test_loss += model.loss(output, target).item()
                cpc = model.get_cpc(data, target)
                test_accuracy += cpc  # only for the last origin of the first batch
                n_origins += 1  #data.shape[1]

            break # only for the first batch
    
        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= n_origins #len(test_sampler)
        test_accuracy /= n_origins #len(test_sampler)
    
        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
    
        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('Test set ({} origins): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                n_origins, test_loss, 100. * test_accuracy))


def evaluate():
    loc2cpc_numerator = {}
    
    model.eval()
    with torch.no_grad():
        
        for [b_data, b_target], ids in test_loader:
            for id, data, target in zip(ids, b_data, b_target):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model.forward(data)
                cpc = model.get_cpc(data, target, numerator_only=True)
                loc2cpc_numerator[id[0]] = cpc

    # gather dictionaries
    comm.barrier()
    ds = comm.gather(loc2cpc_numerator, root=0)
    comm.barrier()
    if rank == 0:
        loc2cpc_numerator = {k:v for d in ds for k,v in d.items()}
        # save to disk
        edf = pd.DataFrame.from_dict(loc2cpc_numerator, columns=['cpc_num'], orient='index')\
                .reset_index().rename(columns={'index': 'locID'})

        # compute average cpc
        oa2tile = {oa:t for t,v in tileid2oa2features2vals.items() for oa in v.keys()}

        def cpc_from_num(edf, oa2tile, o2d2flow):
            edf['tile'] = edf['locID'].apply(lambda x: oa2tile[x])
            edf['tot_flow'] = edf['locID'].apply(lambda x: \
                                               sum(o2d2flow[x].values()) if x in o2d2flow else 0)
            cpc_df = pd.DataFrame(edf.groupby('tile').apply(\
                        lambda x: x['cpc_num'].sum() / 2 / x['tot_flow'].sum()), \
                    columns=['cpc']).reset_index()
            return cpc_df

        cpc_df = cpc_from_num(edf, oa2tile, o2d2flow)
        print(f'Average CPC of test tiles: {cpc_df.cpc.mean():.4f}  stdev: {cpc_df.cpc.std():.4f}')
        fname = wk_dir + 'results/tile2cpc_{}_{}_{}.csv'.format(model_type, country, no_round)
        cpc_df.to_csv(fname, index=False)



if __name__ == '__main__':
    # Run with:

    # CPU
    # mpirun -np 1 python deepgravity_horovod.py --device cpu --momentum 0.9 --lr 1e-5 --epochs 1 --batch_size 10 --test-batch-size 10 >& ../results/log_dg.out

    # GPU
    # mpirun -np 8 python deepgravity_horovod.py --device gpu --seed 42 --momentum 0.9 --lr 5e-6 --epochs 20 --batch_size 64 --test-batch-size 64 --no-round $noround --model-type $modeltype --country $country >& ../results/log_dg_"$modeltype"_"$country"_"$noround".out

    t0 = time.time()
    test()
    for epoch in range(1, args.epochs + 1):
        # set new random seeds
        torch.manual_seed(args.seed + hvd.rank() + epoch)
        np.random.seed(args.seed + hvd.rank() + epoch)
        random.seed(args.seed + hvd.rank() + epoch)
    
        train(epoch)
        test()
    t1 = time.time()
    
    
    if hvd.rank() == 0:
        print("Total training time: %s seconds" %(t1 - t0))
    
        # save model
        fname = wk_dir + 'results/model_{}_{}_{}.pt'.format(model_type, country, no_round)
        print('Saving model to {} ...'.format(fname))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   }, fname)
    
        print('Computing the CPC on test set, loc2cpc_numerator ...')
    
    evaluate()

