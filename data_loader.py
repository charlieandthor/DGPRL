import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import warnings
# from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


def PSMLoader(flag='train'):

    train_data = pd.read_csv("data/PSM/train.csv")
    train_data = train_data.values[:, 1:]
    train_data = np.nan_to_num(train_data)
    test_data = pd.read_csv("data/PSM/test.csv")
    test_data = test_data.values[:, 1:]
    test_data = np.nan_to_num(test_data)
    test_label = pd.read_csv("data/PSM/test_label.csv").values[:, 1:]

    scaler = StandardScaler()
    scaler.fit(train_data)
    train = scaler.transform(train_data)
    test = scaler.transform(test_data)
    val = train[(int)(len(train) * 0.8):]
    # print("train:", train.shape)
    # print("test:", test.shape)

    if flag == "train":
        data = train
    elif flag == "val":
        data = val
    elif flag == "test":
        data = test
    elif flag == "test_label":
        data = test_label
    else:
        raise ValueError("flag must be 'train', 'val', 'test', or 'test_label'")

    return data


def MSLLoader(flag='train'):

    train_data = np.load("data/MSL/MSL_train.npy")
    test_data = np.load("data/MSL/MSL_test.npy")
    test_label = np.load("data/MSL/MSL_test_label.npy")

    scaler = StandardScaler()
    scaler.fit(train_data)
    train = scaler.transform(train_data)
    test = scaler.transform(test_data)
    val = train[(int)(len(train) * 0.8):]
    # print("train:", train.shape)
    # print("test:", test.shape)

    if flag == "train":
        data = train
    elif flag == "val":
        data = val
    elif flag == "test":
        data = test
    elif flag == "test_label":
        data = test_label
    else:
        raise ValueError("flag must be 'train', 'val', 'test', or 'test_label'")

    return data


def SMAPLoader(flag='train'):

    train_data = np.load("data/SMAP/SMAP_train.npy")
    test_data = np.load("data/SMAP/SMAP_test.npy")
    test_label = np.load("data/SMAP/SMAP_test_label.npy")

    scaler = StandardScaler()
    scaler.fit(train_data)
    train = scaler.transform(train_data)
    test = scaler.transform(test_data)
    val = train[(int)(len(train) * 0.8):]
    # print("train:", train.shape)
    # print("test:", test.shape)

    if flag == "train":
        data = train
    elif flag == "val":
        data = val
    elif flag == "test":
        data = test
    elif flag == "test_label":
        data = test_label
    else:
        raise ValueError("flag must be 'train', 'val', 'test', or 'test_label'")

    return data


def SMDLoader(flag='train'):

    train_data = np.load("data/SMD/SMD_train.npy")
    test_data = np.load("data/SMD/SMD_test.npy")
    test_label = np.load("data/SMD/SMD_test_label.npy")

    scaler = StandardScaler()
    scaler.fit(train_data)
    train = scaler.transform(train_data)
    test = scaler.transform(test_data)
    val = train[(int)(len(train) * 0.8):]
    # print("train:", train.shape)
    # print("test:", test.shape)

    if flag == "train":
        data = train
    elif flag == "val":
        data = val
    elif flag == "test":
        data = test
    elif flag == "test_label":
        data = test_label
    else:
        raise ValueError("flag must be 'train', 'val', 'test', or 'test_label'")

    return data


def SWaTLoader(flag='train'):

    train_data = pd.read_csv("data/SWaT/swat_train2.csv")
    train_data = train_data.values[:, :-1]
    test_data = pd.read_csv("data/SWaT/swat2.csv")
    test_label = test_data.values[:, -1:]
    test_data = test_data.values[:, :-1]

    scaler = StandardScaler()
    scaler.fit(train_data)
    train = scaler.transform(train_data)
    test = scaler.transform(test_data)
    val = train[(int)(len(train) * 0.8):]
    # print("train:", train.shape)
    # print("test:", test.shape)

    if flag == "train":
        data = train
    elif flag == "val":
        data = val
    elif flag == "test":
        data = test
    elif flag == "test_label":
        data = test_label
    else:
        raise ValueError("flag must be 'train', 'val', 'test', or 'test_label'")

    return data


