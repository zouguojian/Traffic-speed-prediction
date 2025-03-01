import torch
import numpy as np
import argparse
import time
import util
from trainer import Trainer
from preprocess.model import linear_transformer
import datetime
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='PEMS-BAY', help='')
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data_path', type=str, default='pems_data/', help='data path')
parser.add_argument('--adjdata', type=str, default='pems_data/adjacent.npz', help='adj data path')
parser.add_argument('--input_length', type=int, default=48, help='')
parser.add_argument('--output_length', type=int, default=48, help='')
parser.add_argument('--hid_dim', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=108, help='number of nodes')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--tau', type=int, default=0.25, help='temperature coefficient')
parser.add_argument('--random_feature_dim', type=int, default=64, help='random feature dimension')
parser.add_argument('--node_dim', type=int, default=32, help='node embedding dimension')
parser.add_argument('--time_dim', type=int, default=32, help='time embedding dimension')
parser.add_argument('--time_num', type=int, default=96, help='time in day')
parser.add_argument('--week_num', type=int, default=7, help='day in week')
parser.add_argument('--use_residual', action='store_true', help='use residual connection')
parser.add_argument('--use_bn', action='store_true', help='use batch normalization')
parser.add_argument('--use_spatial', action='store_true', help='use spatial loss')
parser.add_argument('--use_long', action='store_true', help='use long-term preprocessed features')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient cliip')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--milestones', type=list, default=[80, 100], help='optimizer milestones')
parser.add_argument('--patience', type=int, default=30, help='early stopping')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=500, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='checkpoint/', help='save path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/linear_transformer.pth', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()
print(args)


def sample_period(x):
    # trainx (B, N, T, F)
    history_length = x.shape[-2]
    idx_list = [i for i in range(history_length)]
    period_list = [idx_list[i:i + 48] for i in range(0, history_length, args.time_num)]
    period_feat = [x[:, :, sublist, 0] for sublist in period_list]
    period_feat = torch.stack(period_feat)
    period_feat = torch.mean(period_feat, dim=0)

    return period_feat


def main():
    # args.device = torch.device(args.device)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data_path, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(args.device) for i in adj_mx]
    edge_indices = torch.nonzero(supports[0] > 0)

    trainer = Trainer(args, scaler, supports, edge_indices)

    # Default setting: input length 48 output length 48
    # Using long-term features require much more time, it would be faster to pre-compute the features and save them in disk
    if args.use_long:
        feat_extractor = linear_transformer(args.input_length, args.output_length, args.in_dim,
                                            args.num_nodes, args.hid_dim, args.dropout)
        feat_extractor.to(args.device)
        feat_extractor.load_state_dict(torch.load(args.checkpoint))
        for param in feat_extractor.parameters():
            param.requires_grad = False
        feat_extractor.eval()

    print("start testing...", flush=True)
    # trainer.model.load_state_dict(best_state_dict)
    trainer.model.load_state_dict(torch.load('experiments/' + args.name))
    trainer.model.eval()

    y_pred = []
    y_true = []
    start_time = datetime.datetime.now()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(args.device)
        testx = testx.transpose(1, 2) # [B, N, T, D]
        testy = torch.Tensor(y).to(args.device)
        testy = testy.transpose(1, 2)

        if args.use_long:
            feat = []
            for i in range(testx.shape[0]):
                with torch.no_grad():
                    _, feat_sample = feat_extractor(testx[[i], :, :, :])
                feat.append(feat_sample)
            feat = torch.cat(feat, dim=0)
            feat_period = sample_period(testx)
            feat = torch.cat([feat, feat_period], dim=2)
            output, _ = trainer.model(testx, feat)
            # metrics = trainer.eval(testx[:, :, -48:, :], testy[:, :, :, 0], feat, flag='horizon')
        else:
            output, _ = trainer.model(testx, None)
            # metrics = trainer.eval(testx[:, :, -48:, :], testy[:, :, :, 0], flag='horizon')
        y_true.append(testy[:, :, :, 0])
        y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

    np.savez_compressed('data/BigST-' + 'PEMS-BAY',
                        **{'prediction': y_pred.cpu().detach().numpy(), 'truth': y_true.cpu().detach().numpy()})

    print(y_true.shape, y_pred.shape)
    for t in range(y_true.shape[-1]):
        mae = metrics.masked_mae(preds=y_pred[:,:,t], labels=y_true[:,:,t])
        rmse = metrics.masked_rmse(preds=y_pred[:,:,t], labels=y_true[:,:,t])
        mape = metrics.masked_mape(preds=y_pred[:,:,t], labels=y_true[:,:,t])
        print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (t + 1, mae, rmse, mape * 100))


    mae = metrics.masked_mae(preds=y_pred, labels=y_true)
    rmse = metrics.masked_rmse(preds=y_pred, labels=y_true)
    mape = metrics.masked_mape(preds=y_pred, labels=y_true)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' % (mae, rmse, mape * 100))


    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total validating times is : %f" % total_time.total_seconds())


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))