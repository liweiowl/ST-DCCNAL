import numpy as np
import h5py
import pickle
from utils.preprocessing import load_stdata, remove_incomplete_days
from utils.preprocessing import MinMaxNormalization, data_slide_window, MinMaxNormalization_01

data_path = './datasets/BikeNYC/BikeNYC/'


def load_BikeNYC(window_len=6, nb_flow=2, len_test=240):
    # load original data
    data, timestamps = load_stdata(data_path + 'NYC14_M16x8_T60_NewEnd.h5')
    # print(timestamps[:100])

    # remove days that do not have 24 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T=24)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]

    # Min_Max Scale
    data_train = data[:-len_test]
    # print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))
    # save min and max while scaling
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    X, Y = data_slide_window(data=data_all_mmn[0], window_len=window_len)
    xtr, ytr, xte, yte = X[:-len_test], Y[:-len_test], X[-len_test:], Y[-len_test:]
    # print('BikeNYC data loaded...')
    return xtr, ytr, xte, yte


def generate_new_sample(x, T=6):
    nb_sample = len(x)
    x_new = []
    for i in range(T):
        tmp = []
        for j in range(nb_sample):
            tmp.append(x[j][i])
        x_new.append(tmp)
    return x_new


def shuffle_data(x,y):
    permutation = list(np.random.permutation(len(x)))
    x_new = [x[i] for i in permutation]
    y_new = [y[j] for j in permutation]
    return x_new, y_new


def load_BikeNYC_new(window_len=6, nb_flow=2, len_test=240):
    # load original data
    data, timestamps = load_stdata(data_path + 'NYC14_M16x8_T60_NewEnd.h5')
    # print(timestamps[:100])

    # remove days that do not have 24 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T=24)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]

    # Min_Max Scale
    data_train = data[:-len_test]
    # print('train_data shape: ', data_train.shape)
    # mmn = MinMaxNormalization()
    mmn = MinMaxNormalization_01()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))
    # save min and max while scaling
    fpkl = open('preprocessing.pkl', 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    X, Y = data_slide_window(data=data_all_mmn[0], window_len=window_len)
    X, Y = shuffle_data(X, Y)
    # from sklearn.model_selection import train_test_split
    # xtr, ytr, xte, yte = train_test_split(X, Y, test_size=0.1, shuffle=True)
    xtr, ytr, xte, yte = X[:-len_test], Y[:-len_test], X[-len_test:], Y[-len_test:]
    xtr = generate_new_sample(xtr, T=window_len)
    xte = generate_new_sample(xte, T=window_len)
    ytr = np.array(ytr)
    yte = np.array(yte)
    # print('BikeNYC data loaded...')
    return xtr, ytr, xte, yte


if __name__ == '__main__':
    load_BikeNYC()
    load_BikeNYC_new()
    fr = open('preprocessing.pkl', 'rb')
    content = pickle.load(fr)
    fr.close()
    dmax, dmin = content._max, content._min
    print('love world')
