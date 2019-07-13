from __future__ import print_function
import numpy as np
import pandas as pd
from copy import copy
import time
import os
from datetime import datetime, timedelta
import h5py
import platform

np.random.seed(1337)  # for reproducibility


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


class Config(object):
    """docstring for Config"""

    def __init__(self):
        super(Config, self).__init__()

        DATAPATH = os.environ.get('DATAPATH')
        if DATAPATH is None:
            if platform.system() == "Windows" or platform.system() == "Linux":
                # DATAPATH = "D:/data/traffic_flow"
                # elif platform.system() == "Linux":
                DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            else:
                print("Unsupported/Unknown OS: ", platform.system, "please set DATAPATH")
        self.DATAPATH = DATAPATH


def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data, timestamps


"""
    MinMaxNormalization
"""


def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


class MinMaxNormalization_01(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    # vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def split_by_time(data, timestamps, split_timestamp):
    # divide data into two subsets:
    # e.g., Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    assert (len(data) == len(timestamps))
    assert (split_timestamp in set(timestamps))

    data_1 = []
    timestamps_1 = []
    data_2 = []
    timestamps_2 = []
    switch = False
    for t, d in zip(timestamps, data):
        if split_timestamp == t:
            switch = True
        if switch is False:
            data_1.append(d)
            timestamps_1.append(t)
        else:
            data_2.append(d)
            timestamps_2.append(t)
    return (np.asarray(data_1), timestamps_1), (np.asarray(data_2), timestamps_2)


def timeseries2seqs(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i - 1] + offset != timestamps[i]:
            print(timestamps[i - 1], timestamps[i], raw_ts[i - 1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b - 1], breakpoints[b])
        idx = range(breakpoints[b - 1], breakpoints[b])
        for i in range(len(idx) - length):
            x = np.vstack(data[idx[i:i + length]])
            y = data[idx[i + length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def timeseries2seqs_meta(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i - 1] + offset != timestamps[i]:
            print(timestamps[i - 1], timestamps[i], raw_ts[i - 1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    avail_timestamps = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b - 1], breakpoints[b])
        idx = range(breakpoints[b - 1], breakpoints[b])
        for i in range(len(idx) - length):
            avail_timestamps.append(raw_ts[idx[i + length]])
            x = np.vstack(data[idx[i:i + length]])
            y = data[idx[i + length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y, avail_timestamps


def timeseries2seqs_peroid_trend(data, timestamps, length=3, T=48, peroid=pd.DateOffset(days=7), peroid_len=2):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    # timestamps index
    timestamp_idx = dict()
    for i, t in enumerate(timestamps):
        timestamp_idx[t] = i

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i - 1] + offset != timestamps[i]:
            print(timestamps[i - 1], timestamps[i], raw_ts[i - 1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b - 1], breakpoints[b])
        idx = range(breakpoints[b - 1], breakpoints[b])
        for i in range(len(idx) - length):
            # period
            target_timestamp = timestamps[i + length]

            legal_idx = []
            for pi in range(1, 1 + peroid_len):
                if target_timestamp - peroid * pi not in timestamp_idx:
                    break
                legal_idx.append(timestamp_idx[target_timestamp - peroid * pi])
            # print("len: ", len(legal_idx), peroid_len)
            if len(legal_idx) != peroid_len:
                continue

            legal_idx += idx[i:i + length]

            # trend
            x = np.vstack(data[legal_idx])
            y = data[idx[i + length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def timeseries2seqs_3D(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i - 1] + offset != timestamps[i]:
            print(timestamps[i - 1], timestamps[i], raw_ts[i - 1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b - 1], breakpoints[b])
        idx = range(breakpoints[b - 1], breakpoints[b])
        for i in range(len(idx) - length):
            x = data[idx[i:i + length]].reshape(-1, length, 32, 32)
            y = np.asarray([data[idx[i + length]]]).reshape(-1, 1, 32, 32)
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def bug_timeseries2seqs(data, timestamps, length=3, T=48):
    # have a bug
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i - 1] + offset != timestamps[i]:
            breakpoints.append(i)
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b - 1], breakpoints[b])
        idx = range(breakpoints[b - 1], breakpoints[b])
        for i in range(len(idx) - 3):
            x = np.vstack(data[idx[i:i + 3]])
            y = data[idx[i + 3]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def data_slide_window(data, window_len=6):
    x, y = [], []
    for i in range(window_len, len(data)):
        # print(i)
        x.append([data[j] for j in range(i - window_len, i)])
        y.append(data[i])
    return x, y
