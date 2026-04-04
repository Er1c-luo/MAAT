import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


def _build_context(win_size: int) -> np.ndarray:
    # Minimal explicit temporal context: [L, 2] with sin/cos of phase in window
    phase = np.arange(win_size, dtype=np.float32) / np.float32(win_size)
    return np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=-1).astype(np.float32)


def _sin_cos_phase(phase_01: np.ndarray) -> np.ndarray:
    """phase_01: [L] in [0,1) -> [L, 2] with sin/cos(2π phase)."""
    p = np.asarray(phase_01, dtype=np.float32)
    return np.stack([np.sin(2 * np.pi * p), np.cos(2 * np.pi * p)], axis=-1).astype(np.float32)


def _context_from_timestamps_minutes(ts_minutes: np.ndarray, period_min: float = 1440.0) -> np.ndarray:
    """Real-time daily cycle: phase = (timestamp_min mod period) / period -> [L, 2]."""
    ts = np.asarray(ts_minutes, dtype=np.float64)
    phase = (np.mod(ts, period_min) / np.float64(period_min)).astype(np.float32)
    return _sin_cos_phase(phase)


def _context_smd_global_day_index(start_idx: int, win_size: int, period: int = 1440) -> np.ndarray:
    """SMD has no CSV time; use global timestep index mod T (minutes/day) as synthetic day phase."""
    t = np.arange(start_idx, start_idx + win_size, dtype=np.float64)
    phase = (np.mod(t, float(period)) / np.float64(period)).astype(np.float32)
    return _sin_cos_phase(phase)


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        # PSM: align features with timestamp_(min) (or first column) for Version A context
        train_df = pd.read_csv(data_path + '/train.csv')
        if 'timestamp_(min)' in train_df.columns:
            ts_name = 'timestamp_(min)'
        else:
            ts_name = train_df.columns[0]
        self.train_ts = train_df[ts_name].to_numpy(dtype=np.float64)
        feat_train = train_df.drop(columns=[ts_name]).values
        feat_train = np.nan_to_num(feat_train)

        self.scaler.fit(feat_train)
        data = self.scaler.transform(feat_train)

        test_df = pd.read_csv(data_path + '/test.csv')
        if 'timestamp_(min)' in test_df.columns:
            ts_name_te = 'timestamp_(min)'
        else:
            ts_name_te = test_df.columns[0]
        self.test_ts = test_df[ts_name_te].to_numpy(dtype=np.float64)
        test_data = np.nan_to_num(test_df.drop(columns=[ts_name_te]).values)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _context_from_timestamps_minutes(self.train_ts[index:index + self.win_size])
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _context_from_timestamps_minutes(self.test_ts[index:index + self.win_size])
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _context_from_timestamps_minutes(self.test_ts[index:index + self.win_size])
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _context_from_timestamps_minutes(self.test_ts[start:end])
            label = np.float32(self.test_labels[start:end])
            return x, context, label


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[start:end])
            return x, context, label


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[start:end])
            return x, context, label


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self._val_start = int(data_len * 0.8)
        self.val = self.train[self._val_start:]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")
        # Synthetic global time index for test split (after train) for T=1440 phase
        self._test_time_offset = len(self.train)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            # Global day-length cycle T=1440 (minute slots) on series index — no raw timestamps in .npy
            context = _context_smd_global_day_index(index, self.win_size, period=1440)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            g0 = self._val_start + index
            context = _context_smd_global_day_index(g0, self.win_size, period=1440)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            g0 = self._test_time_offset + index
            context = _context_smd_global_day_index(g0, self.win_size, period=1440)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            g0 = self._test_time_offset + start
            context = _context_smd_global_day_index(g0, self.win_size, period=1440)
            label = np.float32(self.test_labels[start:end])
            return x, context, label

class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1
        
    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.flag == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.flag == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[start:end])
            return x, context, label
            
class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[start:end])
            return x, context, label

class NIPS_TS_WaterSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Water_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Water_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Water_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context(self.win_size)
            label = np.float32(self.test_labels[start:end])
            return x, context, label
                                  
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset =='SWAT'):
        dataset = SWATSegLoader(data_path,win_size,1,mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'NIPS_TS_Water'):
        dataset = NIPS_TS_WaterSegLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
