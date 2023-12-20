import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models

#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--traffic_file', default='data/', help='traffic file')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='ningxia-yc', choices=['metr-la', 'ningxia-yc'])
    parser.add_argument('--n_his', type=int, default=12) # 历史输入时长
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--Kt', type=int, default=3) # 时间 fitter 的size
    parser.add_argument('--stblock_num', type=int, default=2) # 层数
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])  # 空间 fitter 的size
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--model_file', default='ningxia-yc',
                        help='save the model to disk')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    # 主要设置每层的输出size
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

def adj_preparate(args, device):
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    return n_vertex

def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def prepare_model(args, blocks, n_vertex):
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return es, model, optimizer, scheduler

def train(args, optimizer, scheduler, es, model, trainX, trainY, valX, valY, mean, std):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    model.train()
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number

        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        for batch_idx in tqdm.tqdm(range(num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

            x = torch.from_numpy(trainX[start_idx: end_idx].astype(np.float)).float().to(device)
            y = torch.from_numpy(trainY[start_idx: end_idx].astype(np.float)).float().to(device)
            y_pred = model(x).view(len(x), -1)   # [batch_size, num_nodes]
            l = _compute_loss(y_pred * std + mean, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item()
            n += y.shape[0]
        val_loss = val(model, valX, valY, mean, std)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.6f} |Train loss: {:.3f} | Val loss: {:.3f} | GPU occupy: {:.3f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / num_batch, val_loss, gpu_mem_alloc))

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model, args.model_file)

        if es.step(val_loss):
            print('Early stopping.')
            break

@torch.no_grad()
def val(model, valX, valY, mean, std):
    model.eval()
    l_sum, n = 0.0, 0
    num_val = valX.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

        x = torch.from_numpy(valX[start_idx: end_idx].astype(np.float)).float().to(device)
        y = torch.from_numpy(valY[start_idx: end_idx].astype(np.float)).float().to(device)
        # y = valY[start_idx: end_idx]

        y_pred = model(x).view(len(x), -1)
        l = _compute_loss(y_pred * std + mean, y)
        l_sum += l.item()
        n += y.shape[0]
    return torch.tensor(l_sum / num_batch)

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

@torch.no_grad() 
def test(mean, std, model, testX, testY, args):
    model = torch.load(args.model_file)
    model.eval()
    pred = []
    label = []
    num_test = testX.shape[0]
    num_batch = math.ceil(num_test / args.batch_size)
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)

            x = torch.from_numpy(testX[start_idx: end_idx].astype(np.float)).float().to(device)
            y = torch.from_numpy(testY[start_idx: end_idx].astype(np.float)).float().to(device)
            # y = testY[start_idx: end_idx]

            y = y.cpu().numpy()
            y_pred = model(x).view(len(x), -1).cpu().numpy()

            pred.append(y_pred)
            label.append(y)
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)
    mae, rmse, mape = metric(pred * std + mean, label)
    # test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, testX, testY, mean, std)
    print(f'Dataset {args.dataset:s} | MAE {mae:.3f} | RMSE {rmse:.3f} | WMAPE {mape:.3f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters() # 参数、设备、每层参数size的定义
    trainX, trainY, valX, valY, testX, testY, mean, std = dataloader.load_dataset(args)
    trainX, valX, testX = np.expand_dims(trainX,axis=1), np.expand_dims(valX,axis=1), np.expand_dims(testX,axis=1)
    trainY, valY, testY = trainY[:,args.n_pred-1], valY[:,args.n_pred-1], testY[:,args.n_pred-1]
    n_vertex = adj_preparate(args, device)

    es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, optimizer, scheduler, es, model, trainX, trainY, valX, valY, mean, std)
    test(mean, std, model, testX, testY, args)