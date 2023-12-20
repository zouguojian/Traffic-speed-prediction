# -- coding: utf-8 --
import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        # >>>需要根据数据集的位置和大小进行特定参数设定
        self.parser.add_argument('--save_path', type=str, default='weights/metr-la/BI-LSTM', help='save path')
        self.parser.add_argument('--model_name', type=str, default='BI-LSTM', help='training or testing model name')
        self.parser.add_argument('--file_train_s', type=str, default='data/metr-la/train_5s.csv', help='training_speed file address')
        self.parser.add_argument('--file_val', type=str, default='data/val.csv', help='validate set file address')
        self.parser.add_argument('--file_test', type=str, default='data/test.csv', help='test set file address')
        self.parser.add_argument('--file_adj', type=str,default='data/metr-la/adjacent.csv', help='adj file address')
        self.parser.add_argument('--site_num', type=int, default=207, help='total number of road')
        self.parser.add_argument('--granularity', type=int, default=5, help='minute granularity')

        # >>>下面参数一般不用动
        self.parser.add_argument('--train_ratio', type=float, default=0.7, help='train data divide')
        self.parser.add_argument('--validate_ratio', type=float, default=0.1, help='validate divide')
        self.parser.add_argument('--test_ratio', type=float, default=0.2, help='test divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epoch', type=int, default=200, help='epoch') # modify
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')  # modify
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.2, help='drop out')
        self.parser.add_argument('--num_heads', type=int, default=8, help='total number of head attentions')
        self.parser.add_argument('--num_blocks', type=int, default=4, help='total number of attention layers')
        self.parser.add_argument('--spatial_top_k', type=int, default=8, help='spatial top k')
        self.parser.add_argument('--temporal_top_k', type=int, default=6, help='temporal top k')
        self.parser.add_argument('--channels', type=int, default=1, help='the number of channels')
        self.parser.add_argument('--predict_steps', type=int, default=1, help='move steps in prediction')

        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layers of lstm')
        self.parser.add_argument('--hidden_size', type=int, default=64, help='hidden size of lstm')

        self.parser.add_argument('--emb_size', type=int, default=64, help='embedding size')
        self.parser.add_argument('--features', type=int, default=1, help='numbers of the feature')
        self.parser.add_argument('--features_p', type=int, default=15, help='numbers of the feature pollution')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=12, help='input length')
        self.parser.add_argument('--output_length', type=int, default=12, help='output length')

        self.parser.add_argument('--training_set_rate', type=float, default=0.7, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.15, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=0.15, help='test set rate')
        self.parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)