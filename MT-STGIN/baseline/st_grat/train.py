from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math
import pandas as pd
import csv

# from lib import utils
from utils import log_string, loadPEMSData
from model import STGRAT

parser = argparse.ArgumentParser()
# parser.add_argument('--time_slot', type = int, default = 5,
#                     help = 'a time step is 5 mins')
parser.add_argument('--cuda', type=int, default=0,
                    help='which gpu card used')
parser.add_argument('--P', type=int, default=12,
                    help='history steps')
parser.add_argument('--Q', type=int, default=6,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=2,
                    help='number of Layers')
parser.add_argument('--k', type=int, default=4,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=1600,
                    help='dims of each head attention outputs')
parser.add_argument('--N', type=float, default=108,
                    help='the node number in graph')  # 节点的数量
parser.add_argument('--K', type=int, default=2,
                    help='control the diffusion steps of K')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.0,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=200,
                    help='epoch to run')
# parser.add_argument('--patience', type = int, default = 10,
#                     help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='initial learning rate')
parser.add_argument('--traffic_file', default='data/we/train.npz',
                    help='traffic file')
parser.add_argument('--SE_file', default='data/we/SE.txt',
                    help='spatial emebdding file')
parser.add_argument('--model_file', default='PEMS',
                    help='save the model to disk')
parser.add_argument('--log_file', default='log(PEMS)',
                    help='log file')
parser.add_argument('--adjacent_file', default='data/we/adjacent.npz',
                    help='adjacent file')
args = parser.parse_args()

log = open(args.log_file, 'w')

# device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

device = torch.device(f"cpu") # 这里需要注意的是,在有GPU的环境下,记得使用上面的语句, 将这一句话注释掉

log_string(log, "loading data....")

trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = loadPEMSData(args)  # 加载数据


# SE = torch.from_numpy(SE).to(device)


def adjecent(file_adj=None):
    '''
    :return: adj matrix
    '''
    data = pd.read_csv(filepath_or_buffer=file_adj)
    adj = np.zeros(shape=[args.N, args.N])
    # for i in range(args.N):adj[i][i]=1
    for line in data[['origin_id', 'destination_id', 'cost']].values:
        adj[int(line[0])][int(line[1])] = line[2]
        # adj[int(line[0])][int(line[1])] = 1

    # adjacent = np.zeros(shape=[args.N*args.N, 3])
    # for i in range(args.N):
    #     for j in range(args.N):
    #         # print(adjacent[i*args.N+j])
    #         adjacent[i*args.N+j,0] = i
    #         adjacent[i * args.N + j, 1] = j
    #         adjacent[i * args.N + j, 2] = adj[i,j]
    # np.savetxt('data/adjacent.txt', adjacent, '%d')
    return adj


# adj = np.array(adjecent(args.adjacent_file),dtype=np.float32)
adj = np.load(args.adjacent_file)['data']  # 加载邻接矩阵
A = torch.from_numpy(adj).to(device)
A = A.float()
D = (A.sum(-1) ** -1)  # 倒数，正则
D[torch.isinf(D)] = 0.
D = torch.diag_embed(D)
A = torch.matmul(D, A)
AS = []
for i in range(args.K):
    AS.append((A ** i))

AT = torch.from_numpy(adj.T).to(device)
AT = AT.float()
DT = (AT.sum(-1) ** -1)
DT[torch.isinf(DT)] = 0.
DT = torch.diag_embed(DT)

AT = torch.matmul(DT, AT)
ATS = []
for i in range(args.K):
    ATS.append((AT ** i))

log_string(log, "loading end....")


def res(model, valX, valTE, valY, mean, std):
    start = time.time()
    model.eval()  # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx: end_idx]).float().to(device)
                y = valY[start_idx: end_idx]
                # te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                encode_state = model.encode(X)
                decode_start = X[:, -1:, :]
                ys = decode_start
                for i in range(y.shape[1]):
                    tmp = model.decode(ys, encode_state)
                    ys = torch.cat([decode_start, tmp], 1)
                    if i == y.shape[1] - 1:
                        y_hat = tmp

                pred.append(y_hat.cpu().numpy() * std + mean)
                label.append(y * std + mean)

    pred = np.concatenate(pred, axis=0)
    label = np.concatenate(label, axis=0)

    y_truth = np.transpose(label, [0, 2, 1]).astype('float')
    y_pred = np.transpose(pred, [0, 2, 1]).astype('float')

    file = open('/Users/guojianzou/Traffic-speed-prediction/MT-STGIN/results/ST-GRAT.csv', 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow( ['road']+ ['label_' + str(i) for i in range(args.Q)] +
        ['predict_' + str(i) for i in range(args.Q)])

    for i in range(y_truth.shape[0]):
        for site in range(args.N):
            writer.writerow([site]+ list(np.round(y_truth[i, site])) + list(np.round(y_pred[i, site])))

    pred = np.transpose(pred, [0, 2, 1]) [:, :28]  # 多任务预测,数据的范围
    label = np.transpose(label, [0, 2, 1])[:, :28]
    print(pred.shape, label.shape)

    maes = []
    rmses = []
    mapes = []
    wapes = []
    rs =[]
    r2s = []

    for i in range(args.Q):
        mae, rmse, mape, wape, r, r2 = metric(pred[:, :, i], label[:, :, i])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        wapes.append(wape)
        rs.append(r)
        r2s.append(r2)
        log_string(log, 'step %d, mae: %.6f, rmse: %.6f, mape: %.6f, wape: %.6f, r: %.6f, r^2: %.6f' % (i + 1, mae, rmse, mape, wape, r, r2))
        # print('step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))

    # segments_index = [32, 34, 47, 50, 51]
    # mae, rmse, mape, wape, r, r2 = metric(pred[:, :, segments_index], label[:, : , segments_index])
    # log_string(log, 'segments, mae: %.6f, rmse: %.6f, mape: %.6f, wape: %.6f, r: %.6f, r^2: %.6f' % (
    # mae, rmse, mape, wape, r, r2))

    mae, rmse, mape, wape, r, r2 = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    wapes.append(wape)
    rs.append(r)
    r2s.append(r2)
    log_string(log, 'average, mae: %.6f, rmse: %.6f, mape: %.6f, wape: %.6f, r: %.6f, r^2: %.6f' % (mae, rmse, mape, wape, r, r2))
    log_string(log, 'eval time: %.1f' % (time.time() - start))
    # print('average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))

    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(rs, 0), np.stack(r2s, 0)


def train(model, trainX, trainTE, trainY, valX, valTE, valY, mean, std):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 43, 46],
                                                        gamma=0.2)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15,
    #                                 verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)

    for epoch in tqdm(range(1, args.max_epoch + 1)):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        # trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(trainX[start_idx: end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx: end_idx]).float().to(device)
                x_t = torch.from_numpy(
                    np.concatenate([trainX[start_idx: end_idx][:, -1:, :], trainY[start_idx: end_idx][:, :-1, :]],
                                   1)).float().to(device)
                # te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)

                optimizer.zero_grad()

                print(X.dtype, x_t.dtype)
                y_hat = model(X, x_t)

                loss = _compute_loss(y * std + mean, y_hat * std + mean)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                train_l_sum += loss.cpu().item()
                # print(f"\nbatch loss: {l.cpu().item()}")
                n += y.shape[0]
                batch_count += 1
                pbar.update(1)
        # lr = lr_scheduler.get_lr()
        log_string(log, 'epoch %d, lr %.6f, loss %.6f, time %.1f sec'
                   % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        # print('epoch %d, lr %.6f, loss %.4f, time %.1f sec'
        #       % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        mae, rmse, mape, _, _ = res(model, valX, valTE, valY, mean, std)
        lr_scheduler.step()
        # lr_scheduler.step(mae[-1])
        if mae[-1] < min_loss:
            min_loss = mae[-1]
            torch.save(model, args.model_file)


def test(model, valX, valTE, valY, mean, std):
    model = torch.load(args.model_file, map_location=device)
    mae, rmse, mape, r, r2 = res(model, valX, valTE, valY, mean, std)
    return mae, rmse, mape, r, r2


def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)

        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')

    return mae, rmse, mape, wape, cor, r2


if __name__ == '__main__':
    maes, rmses, mapes, rs, r2s = [], [], [], [], []
    for i in range(0):
        log_string(log, "model constructed begin....")
        model = STGRAT(SE, 1, args.k * args.d, args.L, args.N, args.k, args.d, args.K, AS, ATS, device).to(device)
        log_string(log, "model constructed end....")
        log_string(log, "train begin....")
        # train(model, trainX, trainTE, trainY, testX, testTE, testY, mean, std)
        log_string(log, "train end....")

    model = STGRAT(SE, 1, args.k * args.d, args.L, args.N, args.k, args.d, args.K, AS, ATS, device).to(device)
    log_string(log, "test begin....")
    mae, rmse, mape, r, r2 = test(model, testX, testTE, testY, mean, std)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    rs.append(r)
    r2s.append(r2)
    log_string(log, "test end....")

    log_string(log, "\n\nresults:")
    maes = np.stack(maes, 1)
    rmses = np.stack(rmses, 1)
    mapes = np.stack(mapes, 1)
    rs = np.stack(rs, 1)
    r2s = np.stack(r2s, 1)
    for i in range(args.Q):
        log_string(log, 'step %d, mae %.6f, rmse %.6f, mape %.6f, r %.6f, r^2 %.6f' % (
        i + 1, maes[i].mean(), rmses[i].mean(), mapes[i].mean(), rs[i].mean(), r2s[i].mean()))
        # log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (i+1, maes[i].std(), rmses[i].std(), mapes[i].std()))
    log_string(log, 'average, mae %.6f, rmse %.6f, mape %.6f, r %.6f, r^2 %.6f' % (maes[-1].mean(), rmses[-1].mean(), mapes[-1].mean(), rs[-1].mean(), r2s[-1].mean()))
    # log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].std(), rmses[-1].std(), mapes[-1].std()))
