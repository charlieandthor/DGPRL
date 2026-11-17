import pandas as pd
import numpy as np
import os
import argparse
import configparser
import warnings
import torch
from copy import deepcopy
import time
import torch.utils.data
import torch.optim as optim
from model import first_model, make_model
import time
import datetime
import math
import torch.nn as nn
from utils import count_parameters, get_normalized_adj, cal_lape, get_adj_from_csv, calculate_normalized_laplacian, get_adj_from_npy, add_data
# from graph_utils import loadGraph
import sklearn
from metrics import print_all_metrics

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser() 
parser.add_argument("--config", default='conf/MSL_1dim_100.conf', type=str,
                     help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(args.config, 'r') as f:
    print(f.read())
print('\n')

config.read(args.config)
data_config = config['Data']
training_config = config['Training']

data_path = data_config['graph_signal_matrix_filename']
distance_csv_path = data_config['distance_csv_path']
dataset_name = data_config['dataset_name']
print("dataset_name: ", dataset_name)
num_of_vertices = int(data_config['num_of_vertices'])


mode = training_config['mode']
ctx = int(training_config['ctx'])
L = int(training_config['L'])
K = int(training_config['K'])
d = int(training_config['d'])
self_weight_dis = float(training_config['self_weight_dis'])
reference = int(training_config['reference'])
learning_rate = float(training_config['learning_rate'])
max_epoch = int(training_config['epochs'])
decay_epoch = int(training_config['decay_epoch'])
batch_size = int(training_config['batch_size'])
num_his = training_config['num_his']
num_pred = int(training_config['num_pred'])
patience = int(training_config['patience'])
in_channels = int(training_config['in_channels'])
lambda_c = float(training_config['lambda_c'])
anomaly_ratio = float(training_config['anmly_ratio'])

# load dataset
data = np.load(data_path)
trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, threX, threTE, testLabel = data['train_x'], data['trainTE'], data['train_target'], data['val_x'], data['valTE'], data['val_target'], data['test_x'], data['testTE'], data['test_target'], data['thre_x'], data['threTE'], data['test_label']
print("train: ", trainX.shape, trainY.shape)
print("val: ", valX.shape, valY.shape)
print("test: ", testX.shape, testY.shape)
print("trainTE: ", trainTE.shape)
print("valTE: ", valTE.shape)
print("testTE: ", testTE.shape)

trainX = torch.from_numpy(np.array(trainX, dtype='float64')).type(torch.FloatTensor)
trainY = torch.from_numpy(np.array(trainY, dtype='float64')).type(torch.FloatTensor)
valX = torch.from_numpy(np.array(valX, dtype='float64')).type(torch.FloatTensor)
valY = torch.from_numpy(np.array(valY, dtype='float64')).type(torch.FloatTensor)
testX = torch.from_numpy(np.array(testX, dtype='float64')).type(torch.FloatTensor)
testY = torch.from_numpy(np.array(testY, dtype='float64')).type(torch.FloatTensor)
threX = torch.from_numpy(np.array(threX, dtype='float64')).type(torch.FloatTensor)
trainTE = torch.from_numpy(np.array(trainTE, dtype='int32'))
valTE = torch.from_numpy(np.array(valTE, dtype='int32'))
testTE = torch.from_numpy(np.array(testTE, dtype='int32'))
threTE = torch.from_numpy(np.array(threTE, dtype='int32'))
testLabel = torch.from_numpy(np.array(testLabel, dtype='int32'))

mean_, std_ = torch.mean(trainX.reshape(-1, in_channels), axis=0), torch.std(trainX.reshape(-1, in_channels), axis=0)
print("mean and std in every feature",mean_.shape, mean_, std_)
trainX = (trainX - mean_) / std_
valX = (valX - mean_) / std_
testX = (testX - mean_) / std_
threX = (threX - mean_) / std_

print("train: ", trainX.shape, trainY.shape)
print("val: ", valX.shape, valY.shape)
print("test: ", testX.shape, testY.shape)
print("trainTE: ", trainTE.shape)
print("valTE: ", valTE.shape)
print("testTE: ", testTE.shape)

gpu = int(training_config['gpu'])
USE_CUDA = torch.cuda.is_available() and gpu >= 0

if USE_CUDA:
    available_gpus = list(range(torch.cuda.device_count()))
    print("Available CUDA GPUs:", available_gpus)

    device = torch.device(f"cuda:{gpu}" if gpu < len(available_gpus) else "cuda:0")
    print("Using device:", device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        multi_gpu = True
    else:
        multi_gpu = False
else:
    device = torch.device("cpu")
    print("Using CPU")
    multi_gpu = False


A, distanceA = get_adj_from_csv(distance_csv_path, num_of_vertices)
lpls = cal_lape(A.copy())
lpls = torch.from_numpy(np.array(lpls, dtype='float32')).type(torch.FloatTensor)
lpls = lpls.to(device)


# set model
model = make_model(config, target_dim=lpls.shape[1], bn_decay=0.1)
# model = make_model(config,localadj,spawave,temwave, bn_decay=0.1)
if multi_gpu:
    model = nn.DataParallel(model)
model = model.to(device)

mean_ = mean_.to(device)
std_ = std_.to(device)

parameters = count_parameters(model)
print('trainable parameters: {:,}'.format(parameters))

# train
print('Start training ...')


val_loss_min = float('inf')
wait = 0
best_model_wts = None
best_model = deepcopy(model.state_dict())
best_epoch = -1
best_loss = np.inf 


num_train = trainX.shape[0]
num_val = valX.shape[0]
num_test = testX.shape[0]
num_thre = threX.shape[0]
train_num_batch = math.ceil(num_train / batch_size)
val_num_batch = math.ceil(num_val / batch_size)
test_num_batch = math.ceil(num_test / batch_size)
thre_num_batch = math.ceil(num_thre / batch_size)


loss_criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=decay_epoch,
                                      gamma=0.9)

exp_datadir = f"experiments/{dataset_name}/"
if not os.path.exists(exp_datadir):
    os.makedirs(exp_datadir)
params_filename = os.path.join(exp_datadir, f"{dataset_name}_{num_of_vertices}_{num_his}_{num_pred}_{K}_{L}_{d}_best_params")

train_time_epochs = []
val_time_epochs=[]
total_start_time = time.time()
for epoch_num in range(0, max_epoch):
    if wait >= patience:
        print(f'early stop at epoch: {epoch_num}, the val loss is {val_loss_min}')
        break
    
    permutation = torch.randperm(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]
    
    start_train = time.time()
    model.train()
    train_loss = 0
    
    print(f"epoch {epoch_num} start!")
    
    for batch_idx in range(train_num_batch):
    
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)
        X = trainX[start_idx: end_idx].to(device)
        # X = add_data(X,number = 2)
        TE = trainTE[start_idx: end_idx].to(device)
        label = trainY[start_idx: end_idx].to(device)
        
        optimizer.zero_grad()
        # pred = model(X,TE,A_Wave, mode)
        # pred = model(X,TE, mode)
        pred, loss_contrast, _ = model(X,TE, lpls, mode)
        # pred = pred * std_[0] + mean_[0]
        loss_batch = loss_contrast
        train_loss += float(loss_batch) * (end_idx - start_idx)
        loss_batch.backward()
        optimizer.step()
        
        if (batch_idx+1) % 10 == 0:
            print(f'Training batch: {batch_idx + 1} in epoch:{epoch_num}, training batch loss:{loss_batch:.4f}')
        del X, TE, label, pred, loss_batch
    train_loss /= num_train
    end_train = time.time()


    print("evaluating on valid set now!")
    val_loss = 0
    start_val = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_val, (batch_idx + 1) * batch_size)
            X = valX[start_idx: end_idx].to(device)
            TE = valTE[start_idx: end_idx].to(device)
            label = valY[start_idx: end_idx].to(device)
            pred, loss_contrast, _ = model(X,TE, lpls, mode, 'test')
            # pred = pred * std_[0] + mean_[0]
            loss_batch = loss_contrast
            val_loss += loss_batch * (end_idx - start_idx)
            del X, TE, label, pred, loss_batch
    val_loss /= num_val
    end_val = time.time()

    train_time_epochs.append(end_train - start_train)
    val_time_epochs.append(end_val - start_val)

    print('%s | epoch: %04d/%d, training time: %.1fs, validation time: %.1fs, lr: %.6f' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch_num + 1,
         max_epoch, end_train - start_train, end_val - start_val, optimizer.param_groups[0]['lr']))
    print(f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

    if val_loss <= val_loss_min:
        wait = 0
        val_loss_min = val_loss
        best_model = deepcopy(model.state_dict())
        best_epoch = epoch_num
    else:
        wait += 1
    scheduler.step()

print("best loss is %.4f at epoch %d", val_loss_min, best_epoch)

print("train one epoch for average: ", np.array(train_time_epochs).mean())
print("valid one epoch for average: ", np.array(val_time_epochs).mean())
print("train time: ", time.time() - total_start_time, "s")

model.load_state_dict(best_model)
model.eval()

print(model)

trainY = trainY.cpu()
valY = valY.cpu()
testY = testY.cpu()
mean_ = mean_.cpu().numpy()
std_ = std_.cpu().numpy()

with torch.no_grad():
    print("Calculating threshold on validation set...")
    val_anomaly_scores = []
    for batch_idx in range(val_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_val, (batch_idx + 1) * batch_size)
        X = valX[start_idx: end_idx].to(device)
        TE = valTE[start_idx: end_idx].to(device)
        _, _, anomaly_score = model(X, TE, lpls, mode, 'test')
        val_anomaly_scores.append(anomaly_score.detach().cpu().numpy())
        del X, TE, anomaly_score

    val_loss_np = np.concatenate(val_anomaly_scores)
    thresh = np.percentile(val_loss_np, 100 - anomaly_ratio)

    print("Running inference on test set...")
    anomaly_scores = []
    start_test = time.time()
    for batch_idx in range(thre_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_thre, (batch_idx + 1) * batch_size)
        X = threX[start_idx: end_idx].to(device)
        TE = threTE[start_idx: end_idx].to(device)
        _, _, anomaly_score = model(X, TE, lpls, mode, 'test')
        anomaly_scores.append(anomaly_score.detach().cpu().numpy())
        del X, TE, anomaly_score

    end_test = time.time()

loss_np = np.concatenate(anomaly_scores)
anmly = (loss_np > thresh).astype(int)
print_all_metrics(testLabel[:anmly.shape[0]], anmly, loss_np)

