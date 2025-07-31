import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class DatasetAlfaAnomaly(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', scale=True, timeenc=0, freq='h'):
        # size [seq_len, pred_len]
        self.flag = flag
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.root_path = root_path
        self.scaler = StandardScaler()
        self.scale = scale
        self.__read_data__()
        

    def __read_data__(self):
        train_data = pd.read_csv(os.path.join(self.root_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(self.root_path, 'test.csv'))
        labels = test_data.values[:, -1:]
        # 删掉训练集和测试集中的故障样本
        train_data = train_data.drop(train_data[(train_data[r'label'] != 0)].index, axis=0)
        # test_data = test_data.drop(test_data[(test_data[r'label'] == 1)].index, axis=0)
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        train_data = train_data.values[:, 1:-1]
        test_data = test_data.values[:, 1:-1]
        data = np.vstack((train_data, test_data))
        

        # if self.scale:
        self.scaler.fit(data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        data_len = len(train_data)
        self.train = train_data[:(int)(data_len * 0.8)]
        self.test = test_data
        self.val = train_data[(int)(data_len * 0.8):]
        self.test_labels = labels
        # print("test:", self.test.shape)
        # print("train:", self.train.shape)
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.flag == "train":
            return np.float32(self.train[s_begin:s_end]), np.float32(self.test_labels[0:self.seq_len]), np.float32(self.train[s_begin:s_end]), np.float32(self.train[r_begin:r_end])
        elif (self.flag == 'val'):
            return np.float32(self.val[s_begin:s_end]), np.float32(self.test_labels[0:self.seq_len]), np.float32(self.val[s_begin:s_end]), np.float32(self.test_labels[s_begin:s_end])
        elif (self.flag == 'test'):
            return np.float32(self.test[s_begin:s_end]), np.float32(self.test_labels[s_begin:s_end]), np.float32(self.test[s_begin:s_end]), np.float32(self.test_labels[s_begin:s_end])

    def __len__(self):
        if self.flag == "train":
            return len(self.train) - self.seq_len - self.pred_len + 1
        elif (self.flag == 'val'):
            return len(self.val) - self.seq_len - self.pred_len + 1
        elif (self.flag == 'test'):
            return len(self.test) - self.seq_len - self.pred_len + 1
        

class DatasetFD(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', scale=True, timeenc=0, freq='h'):
        # size [seq_len, pred_len]
        self.flag = flag
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.root_path = root_path
        self.scaler = StandardScaler()
        self.scale = scale
        self.__read_data__()
        

    def __read_data__(self):
        train_X = pd.read_csv(os.path.join(self.root_path, 'train_X.csv'))
        test_X = pd.read_csv(os.path.join(self.root_path, 'test_X.csv'))
        train_y = pd.read_csv(os.path.join(self.root_path, 'train_y.csv'))
        test_y = pd.read_csv(os.path.join(self.root_path, 'test_y.csv'))
        # 删掉训练集中的故障样本
        fault_index = train_y[train_y['fault'] == 1].index
        train_X = train_X.drop(fault_index, axis=0)
        train_y = train_y.drop(fault_index, axis=0)
        train_X = train_X.values
        train_y = train_y.values
        test_X = test_X.values
        test_y = test_y.values

        # 使用训练集进行归一化
        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X)
        test_X = self.scaler.transform(test_X)
        
        # 选取训练集的0.2作为验证集
        data_len = len(train_X)
        self.train = train_X[:(int)(data_len * 0.8)]
        self.val = train_X[(int)(data_len * 0.8):]
        self.test = test_X
        self.train_labels = train_y[:(int)(data_len * 0.8)]
        self.val_labels = train_y[(int)(data_len * 0.8):]
        self.test_labels = test_y
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.flag == "train":
            return np.float32(self.train[s_begin:s_end]), np.float32(self.test_labels[0:self.seq_len]), np.float32(self.train[s_begin:s_end]), np.float32(self.train[r_begin:r_end])
        elif (self.flag == 'val'):
            return np.float32(self.val[s_begin:s_end]), np.float32(self.test_labels[0:self.seq_len]), np.float32(self.val[s_begin:s_end]), np.float32(self.test_labels[s_begin:s_end])
        elif (self.flag == 'test'):
            return np.float32(self.test[s_begin:s_end]), np.float32(self.test_labels[s_begin:s_end]), np.float32(self.test[s_begin:s_end]), np.float32(self.test_labels[s_begin:s_end])

    def __len__(self):
        if self.flag == "train":
            return len(self.train) - self.seq_len - self.pred_len + 1
        elif (self.flag == 'val'):
            return len(self.val) - self.seq_len - self.pred_len + 1
        elif (self.flag == 'test'):
            return len(self.test) - self.seq_len - self.pred_len + 1
