import argparse
import numpy as np
import os
import sys
import yaml
import torch

from lib.utils import load_graph_data, load_graph_adj
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        if supervisor_config['data'].get('name')=='YINCHUAN':
            sensor_ids, sensor_id_to_ind, adj_mx = load_graph_adj(graph_pkl_filename)
        else: sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        data_type = 'yc'  # 'bay' or 'la'
        supervisor = DCRNNSupervisor(data_type = data_type, LOAD_INITIAL = args.LOAD_INITIAL, adj_mx=adj_mx, **supervisor_config)
        mean_score, outputs = supervisor.evaluate_test('test')
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=True, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/YINCHUAN/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--LOAD_INITIAL', default=True, type=bool, help='If LOAD_INITIAL.')
    parser.add_argument('--TEST_ONLY', default=True, type=bool, help='If TEST_ONLY.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
