# from utils.evaluation_metrics import rmse
from models.CrossLayer_densenet_attentionLSTM import myCrossLayer
from models.iLayer import iLayer
from utils.eval_metric import rmse
from BikeNYC import load_BikeNYC, load_BikeNYC_new
from TaxiBJ import load_data
import numpy as np
from keras import Input, Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import *
from keras.utils.vis_utils import plot_model
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7,0'
start_time = time.time()
import pickle

inf = pickle.load(open('taxi_preprocessing.pkl', 'rb'))
max_flow, min_flow = inf._max, inf._min
print('max_flow is {}, min_flow is {}'.format(max_flow, min_flow))

def cal_TaxiBJ_realRMSE(x):
    return x * (max_flow - min_flow)


def plot_train_val():
    import matplotlib.pyplot as plt
    train = hist.history['root_mean_square_error']
    val = hist.history['val_root_mean_square_error']
    epo = [i for i in range(len(train))]
    plt.xlabel('epoch')
    plt.ylabel('RMSE_loss')
    plt.plot(epo, train, color='green')
    plt.plot(epo, val, color='red')
    plt.scatter(100, score[1], marker='v', c='black')
    plt.legend(['train', 'validation'])
    plt.show()


model_name = './others/{}_best.h5'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
res_all = []
# window_len_list = [2, 3, 4, 6, 9, 12]
window_len_list = [7, 8]
for window_len in window_len_list:
    # window_len = 6
    nb_filter = 4
    filter_size = 3
    nb_layer = 3
    lr = 0.001
    xtr, ytr, xte, yte, external_dim = load_data(window_len=window_len, nb_flow=2, len_test=1304)
    # epochs = 500
    epochs = 100
    batch_size = 256
    # model = build_model(window_len=window_len, lr=lr)
    model = myCrossLayer(nb_flow=2, map_height=32, map_width=32, external_dim=external_dim,
                         nb_layers=nb_layer, window_len=window_len, nb_filter=nb_filter, filter_size=filter_size)
    my_optim = adam(lr=lr)
    model.compile(loss='mse', optimizer=my_optim, metrics=[rmse])
    model.summary()
    plot_model(model, to_file='dense_model.png', show_shapes=True)
    print('hybrid model built ... ')

    # earlystoping = EarlyStopping(monitor='val_root_mean_square_error', patience=30, mode='min')
    earlystoping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    checkpoint = ModelCheckpoint(filepath=model_name, monitor='val_root_mean_square_error',
                                 verbose=1, save_best_only=True, mode='min',
                                 period=1)  # checkpoint2 = ModelCheckpoint(filepath='./others/model_best_2.h5', monitor='val_root_mean_square_error', verbose=1, save_best_only=True,mode='min', period=1)
    print('###' * 10)
    print('xtr length is {}, xtr[0] length is {}'.format(len(xtr), len(xtr[0])))
    hist = model.fit(x=xtr, y=ytr, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
                     callbacks=[earlystoping, checkpoint])
    print('begin to use the best model on validation set to evaluate the test set')
    model.load_weights(model_name)
    score = model.evaluate(x=xte, y=yte, batch_size=32)
    print('raw rmse is', score[1])
    realRMSE = cal_TaxiBJ_realRMSE(score[1])
    print('score[1],the real RMSE norm_01 is ', realRMSE)
    # print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))
    end_time = time.time()
    del_time = end_time - start_time
    print('the whole process use {} minutes'.format(del_time / 60))
    tmp = [window_len, realRMSE]
    res_all.append(tmp)

res_all = np.array(res_all)
print(res_all)
np.save('window_len_7_8.npy', res_all)
# plot_train_val()
print('love world')

