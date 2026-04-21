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
    # Minimal phase context: [L, 2] = [sin(2πp), cos(2πp)]
    phase = np.arange(win_size, dtype=np.float32) / np.float32(win_size)
    return np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=-1).astype(np.float32)


def _build_global_context(start_idx: int, win_size: int, period: int = 1440) -> np.ndarray:
    # Global-time phase context: phase(t) = (global_index mod T) / T, with T=1440 for SMD.
    t = np.arange(start_idx, start_idx + win_size, dtype=np.float32)
    phase = np.mod(t, np.float32(period)) / np.float32(period)
    return np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=-1).astype(np.float32)


def _build_global_multiscale_context(start_idx: int, win_size: int, period1: int = 1440, period2: int = 720) -> np.ndarray:
    # Global-time multiscale context: two periodic views (T1, T2) -> [L, 4].
    t = np.arange(start_idx, start_idx + win_size, dtype=np.float32)
    phase1 = np.mod(t, np.float32(period1)) / np.float32(period1)
    phase2 = np.mod(t, np.float32(period2)) / np.float32(period2)
    return np.stack([
        np.sin(2 * np.pi * phase1), np.cos(2 * np.pi * phase1),
        np.sin(2 * np.pi * phase2), np.cos(2 * np.pi * phase2),
    ], axis=-1).astype(np.float32)


def _build_psm_timestamp_context(ts_min: np.ndarray) -> np.ndarray:
    # PSM explicit time semantics: day (1440 min) + half-day (720 min), output [L, 4].
    ts = np.asarray(ts_min, dtype=np.float32)
    phase_day = np.mod(ts, np.float32(1440.0)) / np.float32(1440.0)
    phase_halfday = np.mod(ts, np.float32(720.0)) / np.float32(720.0)
    return np.stack([
        np.sin(2 * np.pi * phase_day), np.cos(2 * np.pi * phase_day),
        np.sin(2 * np.pi * phase_halfday), np.cos(2 * np.pi * phase_halfday),
    ], axis=-1).astype(np.float32)


# Canonical set of context modes supported across all loaders; all return [L, 4].
CONTEXT_MODES = ("none", "constant", "local_phase", "global_multiscale", "real_timestamp")


def _build_context_by_mode(
    mode: str,
    win_size: int,
    start_idx: int = 0,
    ts_slice: np.ndarray = None,
    period1: int = 1440,
    period2: int = 720,
) -> np.ndarray:
    # Unified ablation switch. All branches keep output shape = [win_size, 4]
    # so the downstream FiLM/context encoder (c_dim=4) stays unchanged.
    if mode == "none":
        # Zero context: isolates the structural effect of the FiLM branch itself.
        return np.zeros((win_size, 4), dtype=np.float32)
    if mode == "constant":
        # Constant (non-zero) context: tests whether any non-trivial input suffices.
        return np.ones((win_size, 4), dtype=np.float32)
    if mode == "local_phase":
        # Window-local phase (no global index, no timestamp): weakest periodic signal.
        phase = np.arange(win_size, dtype=np.float32) / np.float32(win_size)
        phase_half = np.mod(2.0 * phase, 1.0)
        return np.stack([
            np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase),
            np.sin(2 * np.pi * phase_half), np.cos(2 * np.pi * phase_half),
        ], axis=-1).astype(np.float32)
    if mode == "real_timestamp":
        # Honor real timestamps only when the caller provides a ts slice (e.g. PSM).
        # Other datasets transparently fall back to global_multiscale to keep shape.
        if ts_slice is not None:
            return _build_psm_timestamp_context(ts_slice)
        return _build_global_multiscale_context(start_idx, win_size, period1, period2)
    # Default: global_multiscale (day + halfday from the global index).
    return _build_global_multiscale_context(start_idx, win_size, period1, period2)


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", context_mode="global_multiscale"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.context_mode = context_mode
        self.scaler = StandardScaler()
        train_df = pd.read_csv(data_path + '/train.csv')
        # Reuse timestamp_(min) when available; otherwise fallback to the first column as time index.
        ts_col = 'timestamp_(min)' if 'timestamp_(min)' in train_df.columns else train_df.columns[0]
        self.train_ts_min = pd.to_numeric(train_df[ts_col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        data = train_df.drop(columns=[ts_col]).values
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_df = pd.read_csv(data_path + '/test.csv')
        ts_col_test = 'timestamp_(min)' if 'timestamp_(min)' in test_df.columns else test_df.columns[0]
        self.test_ts_min = pd.to_numeric(test_df[ts_col_test], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        test_data = test_df.drop(columns=[ts_col_test]).values
        test_data = np.nan_to_num(test_data)

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
        # PSM is the only dataset with a true timestamp column; pass the slice through
        # so `real_timestamp` mode actually uses it, while other modes ignore it.
        train_offset = len(self.train)
        if self.mode == "train":
            x = np.float32(self.train[index:index + self.win_size])
            context = _build_context_by_mode(
                self.context_mode, self.win_size, start_idx=index,
                ts_slice=self.train_ts_min[index:index + self.win_size],
            )
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context_by_mode(
                self.context_mode, self.win_size, start_idx=train_offset + index,
                ts_slice=self.test_ts_min[index:index + self.win_size],
            )
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context_by_mode(
                self.context_mode, self.win_size, start_idx=train_offset + index,
                ts_slice=self.test_ts_min[index:index + self.win_size],
            )
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context_by_mode(
                self.context_mode, self.win_size, start_idx=train_offset + start,
                ts_slice=self.test_ts_min[start:end],
            )
            label = np.float32(self.test_labels[start:end])
            return x, context, label


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", context_mode="global_multiscale"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.context_mode = context_mode
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        # Treat test split as continuing global time after train for index-based context.
        self._test_time_offset = len(self.train)
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
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + index)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + start)
            label = np.float32(self.test_labels[start:end])
            return x, context, label


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", context_mode="global_multiscale"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.context_mode = context_mode
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        # Treat test split as continuing global time after train for index-based context.
        self._test_time_offset = len(self.train)
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
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + index)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + start)
            label = np.float32(self.test_labels[start:end])
            return x, context, label


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", context_mode="global_multiscale"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.context_mode = context_mode
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
        # Treat test split as continuing global time after train.
        self._test_time_offset = len(self.train)
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

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
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._val_start + index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.mode == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + index)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + start)
            label = np.float32(self.test_labels[start:end])
            return x, context, label

class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train", context_mode="global_multiscale"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.context_mode = context_mode
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
        self._val_start = int(data_len * 0.8)
        self.val = self.train[self._val_start:]
        # Treat test split as continuing global time after train for index-based context.
        self._test_time_offset = len(self.train)
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
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.flag == 'val'):
            x = np.float32(self.val[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._val_start + index)
            label = np.float32(self.test_labels[0:self.win_size])
            return x, context, label
        elif (self.flag == 'test'):
            x = np.float32(self.test[index:index + self.win_size])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + index)
            label = np.float32(self.test_labels[index:index + self.win_size])
            return x, context, label
        else:
            start = index // self.step * self.win_size
            end = start + self.win_size
            x = np.float32(self.test[start:end])
            context = _build_context_by_mode(self.context_mode, self.win_size, start_idx=self._test_time_offset + start)
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
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

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
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
                                  
def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD',
                       context_mode='global_multiscale'):
    # `context_mode` is forwarded to loaders that emit (x, context, label).
    # NIPS_TS_* still emit (x, label); their context is produced by solver._fallback_context.
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode, context_mode=context_mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode, context_mode=context_mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode, context_mode=context_mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode, context_mode=context_mode)
    elif (dataset =='SWAT'):
        dataset = SWATSegLoader(data_path, win_size, 1, mode, context_mode=context_mode)
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
