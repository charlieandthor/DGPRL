import numpy as np
import os
import pandas as pd
import argparse
import configparser
import warnings
import datetime
import re
# from statsmodels.tsa.stattools import grangercausalitytests

from data_loader import PSMLoader, MSLLoader, SMAPLoader, SMDLoader, SWaTLoader


def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his  + 1
    x = np.zeros(shape = (num_sample, num_his, dims))
    y = np.zeros(shape = (num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i: i + num_pred]
    return x, y

# def seq2instance_plus(data, num_his, num_pred):
#     num_step = data.shape[0]
#     num_sample = num_step - num_his - num_pred + 1
#     x = []
#     y = []
#     for i in range(num_sample):
#         x.append(data[i: i + num_his])
#         y.append(data[i + num_his: i + num_his + num_pred, :, :1])
#     x = np.array(x)
#     y = np.array(y)
#     return x, y

def seq2instance_plus(data, win_size):
    num_step, dims = data.shape
    num_sample = num_step // win_size
    x = np.zeros(shape = (num_sample, win_size, dims))
    # y = np.zeros(shape = (num_sample, win_size, dims))
    for i in range(num_sample, win_size):
        x[i] = data[i: i + win_size]
        # y[i] = data[i: i + win_size]
    return x

def granger_binary_graph(ts, max_lag=5, p_th=0.05):
    """
    input:
        df: (T, N) DataFrame
    output:
        edge_df: DataFrame with columns ["from", "to", "edge"]，edge = 0/1
    """
    # ts = df.values
    N = ts.shape[1]
    print(N)
    edges = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # column_stack([Y, X])
            # X = i, Y = j
            data = np.column_stack([ts[:, j], ts[:, i]])
            try:
                result = grangercausalitytests(
                    data, maxlag=max_lag, verbose=False
                )
                test_res = result[max_lag][0]
                # p-value of SSR F-test
                p_value = test_res['ssr_ftest'][1]
                # 1/0
                if p_value < p_th:
                    edges.append([i, j, 1])
            except Exception:
                # edges.append([i, j, 0])
                continue

    return pd.DataFrame(edges, columns=["from", "to", "value"])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='conf/MSL_construct_samples.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    # training_config = config['Training']

    # time_slice_size = int(data_config['time_slice_size'])
    # train_ratio = float(data_config['train_ratio'])
    # val_ratio = float(data_config['val_ratio'])
    # test_ratio = float(data_config['test_ratio'])
    dataset_name = data_config['dataset_name']
    num_his = int(data_config['num_his'])
    num_pred = int(data_config['num_pred'])

    data_dict = {
        'PSM': PSMLoader,
        'MSL': MSLLoader,
        'SMAP': SMAPLoader,
        'SMD': SMDLoader,
        'SWAT': SWaTLoader
    }

    train_data = data_dict[dataset_name](flag="train")
    val_data = data_dict[dataset_name](flag="val")
    test_data = data_dict[dataset_name](flag="test")
    test_labels = data_dict[dataset_name](flag="test_label")
    num_of_vertices = train_data.shape[1]
    print('num of vertices ', num_of_vertices)

    sets = {'train': train_data, 'val': val_data, 'test': test_data}
    xy = {}
    te = {}
    for set_name in sets.keys():
        data_set = sets[set_name].astype("float64")
        X, Y = seq2instance(data_set, num_his, num_pred)
        xy[set_name] = [X, Y]

        # total samples
        total_steps = data_set.shape[0]
        # For each timestamp t, create step index [t, 0]
        # shape = (total_steps, 2)
        total_ids = np.arange(total_steps).reshape(-1, 1)
        time = np.concatenate((total_ids, total_ids[::-1]), axis=-1)
        # seq2instance will cut it into windows
        time = seq2instance(time, num_his, num_pred)
        # Combine history and future
        te[set_name] = np.concatenate(time, 1).astype(np.int32)

    x_trains, y_trains = xy['train']
    x_vals, y_vals = xy['val']
    x_tests, y_tests = xy['test']

    trainTEs = te['train']
    valTEs = te['val']
    testTEs = te['test']

    x_thre = seq2instance_plus(test_data, num_his)
    total_ids = np.arange(test_data.shape[0]).reshape(-1, 1)
    time = np.concatenate((total_ids, total_ids[::-1]), axis=-1)
    threTEs = seq2instance_plus(time, num_his)
    threTEs = np.concatenate([threTEs, threTEs], 1).astype(np.int32)

    output_dir = data_config['output_dir']
    output_path = os.path.join(output_dir, "samples_" + str(num_his) + "_" + str(num_pred) + ".npz")
    print(f"save file to {output_path}")
    np.savez_compressed(
            output_path,
            train_x=x_trains, train_target=y_trains,
            val_x=x_vals, val_target=y_vals,
            test_x=x_tests, test_target=y_tests,
            trainTE=trainTEs, testTE=testTEs, valTE=valTEs,
            thre_x=x_thre, threTE=threTEs, test_label=test_labels)


    # edge_df = granger_binary_graph(train_data)
    # edge_df.to_csv("granger_edges_binary.csv", index=False)
    #
    # print("完成！输出文件：granger_edges_binary.csv")
    # print(edge_df.head())

