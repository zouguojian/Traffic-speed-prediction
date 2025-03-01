import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import metrics
from model import BigST

class Trainer():
    def __init__(self, args, scaler, supports, edge_indices):
        self.model = BigST(args, supports, edge_indices)
        self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1, verbose=False)
        
        self.loss = metrics.masked_mae
        self.scaler = scaler
        self.use_spatial = args.use_spatial
        self.grad_clip = args.grad_clip

    def train(self, input, real_val, feat=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_spatial:
            output, spatial_loss = self.model(input, feat)
            real = real_val
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)-0.3*spatial_loss
        else:
            output, _ = self.model(input, feat)
            real = real_val
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
        
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        mape = metrics.masked_mape(predict,real,0.0).item()
        rmse = metrics.masked_rmse(predict,real,0.0).item()
        return loss.item(), mape, rmse
    
    def eval(self, input, real_val, feat=None, flag='overall'):
        if flag=='overall':
            self.model.eval()
            output, _ = self.model(input, feat)
            real = real_val
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
            mape = metrics.masked_mape(predict,real,0.0).item()
            rmse = metrics.masked_rmse(predict,real,0.0).item()
            return loss.item(), mape, rmse
        elif flag=='horizon':
            self.model.eval()
            output, _ = self.model(input, feat)
            real = real_val
            predict = self.scaler.inverse_transform(output)
            loss = []
            mape = []
            rmse = []
            for i in range(48):
                loss.append(self.loss(predict[..., i], real[..., i], 0.0).item())
                mape.append(metrics.masked_mape(predict[..., i], real[..., i], 0.0).item())
                rmse.append(metrics.masked_rmse(predict[..., i], real[..., i], 0.0).item())
            return loss, mape, rmse
