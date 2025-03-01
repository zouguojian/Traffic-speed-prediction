import torch
import numpy as np
import argparse
import time
import util
import copy
from trainer import Trainer
from preprocess.model import linear_transformer

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='PEMS-BAY',help='')
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_path',type=str,default='pems_data/',help='data path')
parser.add_argument('--adjdata',type=str,default='pems_data/adjacent.npz',help='adj data path')
parser.add_argument('--input_length',type=int,default=48,help='')
parser.add_argument('--output_length',type=int,default=48,help='')
parser.add_argument('--hid_dim',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=108,help='number of nodes')
parser.add_argument('--num_layers',type=int,default=3,help='number of layers')
parser.add_argument('--tau',type=int,default=0.25,help='temperature coefficient')
parser.add_argument('--random_feature_dim',type=int,default=64,help='random feature dimension')
parser.add_argument('--node_dim',type=int,default=32,help='node embedding dimension')
parser.add_argument('--time_dim',type=int,default=32,help='time embedding dimension')
parser.add_argument('--time_num',type=int,default=96,help='time in day')
parser.add_argument('--week_num',type=int,default=7,help='day in week')
parser.add_argument('--use_residual', action='store_true', help='use residual connection')
parser.add_argument('--use_bn', action='store_true', help='use batch normalization')
parser.add_argument('--use_spatial', action='store_true', help='use spatial loss')
parser.add_argument('--use_long', action='store_true', help='use long-term preprocessed features')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--grad_clip',type=float,default=5,help='gradient cliip')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--milestones',type=list,default=[80, 100],help='optimizer milestones')
parser.add_argument('--patience',type=int,default=30,help='early stopping')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=500,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--save',type=str,default='checkpoint/',help='save path')
parser.add_argument('--checkpoint',type=str,default='checkpoint/linear_transformer.pth',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()
print(args)

def sample_period(x):
    # trainx (B, N, T, F)
    history_length = x.shape[-2]
    idx_list = [i for i in range(history_length)]
    period_list = [idx_list[i:i+48] for i in range(0, history_length, args.time_num)]
    period_feat = [x[:,:,sublist,0] for sublist in period_list]
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
    
    print("start training...",flush=True)
    his_loss =[]
    test_time = []
    val_time = []
    train_time = []

    wait = 0
    min_val_loss = np.inf
    
    for i in range(1, args.epochs+1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # trainx: long history data T=2016 or 864
            trainx = torch.Tensor(x).to(args.device) # (B, T, N, F)
            trainx = trainx.transpose(1, 2) # (B, N, T, F)
            trainy = torch.Tensor(y).to(args.device) # (B, T, N, F)
            trainy = trainy.transpose(1, 2) # (B, N, T, F)
            if args.use_long:
                feat = []
                for i in range(trainx.shape[0]):
                    with torch.no_grad():
                        _, feat_sample = feat_extractor(trainx[[i],:,:,:])
                    feat.append(feat_sample)
                feat = torch.cat(feat, dim=0)
                feat_period = sample_period(trainx)
                feat = torch.cat([feat, feat_period], dim=2)
                # trainx[:,:,-48:,:]: short history data T=48
                metrics = trainer.train(trainx[:,:,-48:,:], trainy[:,:,:,0], feat)
            else:
                # trainx[:,:,-48:,:]: short history data T=48
                metrics = trainer.train(trainx[:,:,-48:,:], trainy[:,:,:,0])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            t2 = time.time()
            train_time.append(t2-t1)
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        trainer.scheduler.step()
        
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(args.device)
            testx = testx.transpose(1, 2)
            testy = torch.Tensor(y).to(args.device)
            testy = testy.transpose(1, 2)
            
            if args.use_long:
                feat = []
                for i in range(testx.shape[0]):
                    with torch.no_grad():
                        _, feat_sample = feat_extractor(testx[[i],:,:,:])
                    feat.append(feat_sample)
                feat = torch.cat(feat, dim=0)
                feat_period = sample_period(testx)
                feat = torch.cat([feat, feat_period], dim=2)
                metrics = trainer.eval(testx[:,:,-48:,:], testy[:,:,:,0], feat)
            else:
                metrics = trainer.eval(testx[:,:,-48:,:], testy[:,:,:,0])
                
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        if mvalid_loss < min_val_loss:
            wait = 0
            min_val_loss = mvalid_loss
            # best_epoch = i
            # best_state_dict = copy.deepcopy(trainer.model.state_dict())
            torch.save(trainer.model.state_dict(), 'experiments/' + args.name)
        else:
            wait += 1
            if wait >= args.patience:
                break
        
        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)

    # test
    # trainer.model.load_state_dict(best_state_dict)
    trainer.model.load_state_dict(torch.load('experiments/' + args.name))
    trainer.model.eval()
    test_loss = {str(i):[] for i in range(48)}
    test_mape = {str(i):[] for i in range(48)}
    test_rmse = {str(i):[] for i in range(48)}
    s1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(args.device)
        testx = testx.transpose(1, 2)
        testy = torch.Tensor(y).to(args.device)
        testy = testy.transpose(1, 2)
        
        if args.use_long:
            feat = []
            for i in range(testx.shape[0]):
                with torch.no_grad():
                    _, feat_sample = feat_extractor(testx[[i],:,:,:])
                feat.append(feat_sample)
            feat = torch.cat(feat, dim=0)
            feat_period = sample_period(testx)
            feat = torch.cat([feat, feat_period], dim=2)
            metrics = trainer.eval(testx[:,:,-48:,:], testy[:,:,:,0], feat, flag='horizon')
        else:
            metrics = trainer.eval(testx[:,:,-48:,:], testy[:,:,:,0], flag='horizon')
            
        for k in range(48):
            test_loss[str(k)].append(metrics[0][k])
            test_mape[str(k)].append(metrics[1][k])
            test_rmse[str(k)].append(metrics[2][k])
    s2 = time.time()
    log = 'Epoch: {:03d}, Test Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    test_time.append(s2-s1)
    amae = []
    amape = []
    armse = []
    for k in range(48):
        amae.append(np.mean(test_loss[str(k)]))
        amape.append(np.mean(test_mape[str(k)]))
        armse.append(np.mean(test_rmse[str(k)]))
        log = 'Model performance for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(k+1, amae[-1], amape[-1], armse[-1]))

    log = 'On average over 48 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))

