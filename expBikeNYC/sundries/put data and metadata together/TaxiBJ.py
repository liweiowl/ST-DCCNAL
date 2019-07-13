# -*- coding: utf-8 -*-
from __future__ import print_function
import os
# import cPickle as pickle
import pickle
from copy import copy
import numpy as np
import h5py
from utils.preprocessing import load_stdata, STMatrix
from utils.preprocessing import MinMaxNormalization, remove_incomplete_days, timestamp2vec

# from ..config import Config
np.random.seed(1337)  # for reproducibility

# parameters
# DATAPATH = Config().DATAPATH
datapath = './datasets/TaxiBJ/'


def data_slide_window_timestamps(data, timestamps, meta_data, window_len=6):
    x, y, timestamps_X, meta_data_X = [], [], [], []
    for i in range(window_len, len(data)):
        # print(i)
        x.append([data[j] for j in range(i - window_len, i)])
        timestamps_X.append([timestamps[j] for j in range(i - window_len, i)])
        meta_data_X.append([meta_data[j] for j in range(i - window_len, i)])
        y.append(data[i])
    return x, y, timestamps_X, meta_data_X


def data_slide_window(data, window_len=6):
    x, y = [], []
    for i in range(window_len, len(data)):
        # print(i)
        x.append([data[j] for j in range(i - window_len, i)])
        y.append(data[i][0])
    return x, y


def load_holiday(timeslots, fname=os.path.join(datapath, 'TaxiBJ', 'BJ_Holiday.txt')):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])  # delete the blank of every string
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]


# holiday_test = load_holiday(timeslots=24)


def load_meteorol(timeslots, fname=os.path.join(datapath, 'TaxiBJ', 'BJ_Meteorology.h5')):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


# def load_data(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
#               len_test=None, preprocess_name='preprocessing.pkl',
#               meta_data=True, meteorol_data=True, holiday_data=True):
def load_data(T=48, nb_flow=2, len_test=None, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=True, holiday_data=True, window_len=12):
    # assert(len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            datapath, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        # stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    data_all_mmn_vstack = np.vstack(copy(data_all_mmn))
    timestamps_all_vstack = []
    for timestamps_element in timestamps_all:
        timestamps_all_vstack += timestamps_element
    # timestamps_all_vstack = np.vstack(copy(timestamps_all)

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_all_vstack)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_all_vstack)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_all_vstack)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    data_with_metadata = [[data_all_mmn_vstack[i], meta_feature[i]] for i in range(len(data_all_mmn_vstack))]
    # X, Y, timestamps_X, meta_feature_X = data_slide_window_timestamps(data_all_mmn_vstack, window_len=window_len,
    #                                                                timestamps=timestamps_all_vstack,
    #                                                                meta_data=meta_feature)
    X, Y, = data_slide_window(data_with_metadata, window_len=window_len)
    # s = shuffle_data_many([X, Y, timestamps_X, meta_feature_X])
    s = shuffle_data_many([X, Y])
    X, Y = s[0], s[1]
    # X, Y, timestamps_X, meta_feature = s[0], s[1], s[2], s[3]
    X_train, X_test = X[:-len_test], X[-len_test:]
    Y_train, Y_test = Y[:-len_test], Y[-len_test:]

    # if metadata_dim is not None:
    #     meta_feature_train, meta_feature_test = meta_feature[
    #                                             :-len_test], meta_feature[-len_test:]
    #
    # X_train.append(meta_feature_train)
    # X_test.append(meta_feature_test)

    return X_train, Y_train, X_test, Y_test, metadata_dim


def generate_new_sample(x, T=6):
    nb_sample = len(x)
    x_new = []
    for i in range(T):
        tmp = []
        for j in range(nb_sample):
            tmp.append(x[j][i])
        x_new.append(tmp)
    return x_new


def shuffle_data_many(a):
    permutation = list(np.random.permutation(len(a[0])))
    a_shuffle = []
    for x in a:
        x_new = [x[i] for i in permutation]
        a_shuffle.append(x_new)
    return a_shuffle


(T, nb_flow, len_test) = (48, 2, 1344)
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_test=len_test, preprocess_name='preprocessing.pkl',
        meta_data=True, meteorol_data=True, holiday_data=True)

    print('love world')
