# -- coding: utf-8 --

import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        self.parser.add_argument('--save_path', type=str, default='gcn/', help='save path')
        self.parser.add_argument('--model_name', type=str, default='lstm', help='training or testing model name')

        self.parser.add_argument('--divide_ratio', type=float, default=0.8, help='data_divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epoch', type=int, default=50, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.3, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=66, help='total number of road')
        self.parser.add_argument('--num_heads', type=int, default=4, help='total number of head attentions')
        self.parser.add_argument('--num_blocks', type=int, default=1, help='total number of attention layers')

        #每个点表示一个监测站点，目前66个监测站点
        self.parser.add_argument('--emb_size', type=int, default=256, help='embedding size')
        self.parser.add_argument('--features', type=int, default=1, help='numbers of the feature')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=4, help='input length')
        self.parser.add_argument('--output_length', type=int, default=2, help='output length')

        self.parser.add_argument('--hidden1', type=int, default=32, help='number of units in hidden layer 1')
        self.parser.add_argument('--gcn_output_size', type=int, default=64, help='model string')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')

        self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=0.7, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.15, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=0.15, help='test set rate')

        self.parser.add_argument('--file_train', type=str, default='data/train.csv',help='training set address')
        self.parser.add_argument('--file_val', type=str, default='data/val.csv', help='validation set address')
        self.parser.add_argument('--file_test', type=str, default='data/test.csv', help='test set address')

        self.parser.add_argument('--file_adj', type=str, default='data/adjacent_fully.csv',
                                 help='adj file address')

        self.parser.add_argument('--file_out', type=str, default='ckpt', help='file out')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)