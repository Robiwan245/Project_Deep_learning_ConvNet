import numpy as np 
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from tqdm.notebook import tqdm
from tqdm import tqdm
import pickle

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

        X, _, _ = normalize_data(data[b"data"].T)
        y = np.array(data[b"labels"])
        Y = np.eye(10)[y].T
    return X, Y, y

def normalize_data(X, axis=1, keepdims=True):
    X_mean = np.mean(X, axis=axis, keepdims=keepdims)
    X_std = np.std(X, axis=axis, keepdims=keepdims)
    X = (X - X_mean) / X_std
    return X, X_mean, X_std

def unpickle(filename):
    with open(filename, 'rb') as f:
        file_dict = pickle.load(f, encoding='bytes')
    return file_dict

def load_all_data(val_offset):

    filename_1 = "datasets/cifar-10-batches-py/data_batch_1"
    filename_2 = "datasets/cifar-10-batches-py/data_batch_2"
    filename_3 = "datasets/cifar-10-batches-py/data_batch_3"
    filename_4 = "datasets/cifar-10-batches-py/data_batch_4"
    filename_5 = "datasets/cifar-10-batches-py/data_batch_5"
    test_file = "datasets/cifar-10-batches-py/test_batch"

    X_train1, Y_train1, y_train1 = load_data(filename_1)
    X_train2, Y_train2, y_train2 = load_data(filename_2)
    X_train3, Y_train3, y_train3 = load_data(filename_3)
    X_train4, Y_train4, y_train4 = load_data(filename_4)
    X_train5, Y_train5, y_train5 = load_data(filename_5)

    X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5), axis=1)
    Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5), axis=1)
    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
    X_val = X_train[:, -val_offset:]
    Y_val = Y_train[:, -val_offset:]
    y_val = y_train[-val_offset:]
    X_train = X_train[:, :-val_offset]
    Y_train = Y_train[:, :-val_offset]
    y_train = y_train[:-val_offset]

    X_test, Y_test, y_test = load_data(test_file)

    labels = unpickle('datasets/cifar-10-batches-py/batches.meta')[ b'label_names']

    data = {'X_train': X_train,
            'Y_train': Y_train,
            'y_train': y_train,
            'X_val': X_val,
            'Y_val': Y_val,
            'y_val': y_val,
            'X_test': X_test,
            'Y_test': Y_test,
            'y_test': y_test}

    return data, labels

trainData = pd.read_csv('cifar-10/trainLabels.csv')
trainData.head()

print("Number of points:",trainData.shape[0])
print("Number of features:",trainData.shape[1])
print("Features:",trainData.columns.values)
print("Number of Unique Values")
for col in trainData:
    print(col,":",len(trainData[col].unique()))
plt.figure(figsize=(12,8))