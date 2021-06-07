# -- coding: utf-8 --

import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        self.parser.add_argument('--target_site_id', type=int, default=0, help='city ID')
        self.parser.add_argument('--data_divide', type=float, default=0.9, help='data_divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epochs', type=int, default=200, help='epoch')
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=47, help='total number of city')

        #每个点表示a->b路线，目前8个收费站
        self.parser.add_argument('--features', type=int, default=7, help='numbers of the feature')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=3, help='input length')
        self.parser.add_argument('--output_length', type=int, default=1, help='output length')

        self.parser.add_argument('--model_name', type=str, default='gcn', help='model string')
        self.parser.add_argument('--hidden1', type=int, default=16, help='number of units in hidden layer 1')
        self.parser.add_argument('--gcn_output_size', type=int, default=32, help='model string')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')

        self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=1.0, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.0, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=1.0, help='test set rate')

        self.parser.add_argument('--file_train', type=str,
                                 default='data/train_around_weather.csv',
                                 help='training set file address')
        self.parser.add_argument('--file_val', type=str,
                                 default='/Users/guojianzou/Documents/program/shanghai_weather/val_around_weather.csv',
                                 help='validate set file address')
        self.parser.add_argument('--file_test', type=str,
                                 default='data/around_weathers_2017_7_test.csv',
                                 help='test set file address')

        self.parser.add_argument('--file_adj', type=str,
                                 default='data/adjacent.csv',
                                 help='adj file address')

        self.parser.add_argument('--file_out', type=str, default='ckpt', help='file out')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)