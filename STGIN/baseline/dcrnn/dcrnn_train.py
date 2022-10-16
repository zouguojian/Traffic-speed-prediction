from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import yaml

from lib.utils import load_graph_data,load_adjacent
from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    '''
    仅用做模型的定义，和邻接矩阵的传入工作
    :param args:
    :return:
    '''
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)
        #
        # graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        # sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        adj_mx = load_adjacent(supervisor_config['data'].get('adjacent_dir'))
        '''
        sensor_ids: 表示的原始站点名字，列表形式，如 ['400001', '400017', '400030',...,]
        ensor_id_to_ind: 表示原始名字到index上的映射，字典形式，如{'401391': 148, '409528': 317, '407157': 263,...,}
        adj_mx: 表示邻接矩阵，对于流量可以用距离来表示，但是对于速度，邻接矩阵只能用0，1表示，numpy类型的数据
        '''

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
            supervisor.train(sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
