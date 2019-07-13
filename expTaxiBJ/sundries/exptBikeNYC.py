# from utils.evaluation_metrics import rmse
from models.HA_model import myHA
from models.ARIMA_model import myARIMA
from models.CrossLayer_model_dense import myCrossLayer
from models.iLayer import iLayer
from utils.eval_metric import rmse
from BikeNYC import load_BikeNYC, load_BikeNYC_new
import numpy as np
from keras import Input, Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import *
from keras.utils.vis_utils import plot_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def cal_real_rmse(x):
    m_factor = (16 * 8 / 81) ** 0.5
    return x / 2 * 267 * m_factor


def cal_real_rmse_01norm(x):
    m_factor = (16 * 8 / 81) ** 0.5
    return x * 267 * m_factor


def HA_test(xtr=None, ytr=None, xte=[], yte=[], window_len=1):
    # when the slide window length is 1, we get the minimum rmse: 11.763420450683387
    pred = myHA(xte=xte, yte=yte)
    sum = 0
    N = len(pred) * len(pred[0]) * len(pred[0][0]) * len(pred[0][0][0])
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            for k in range(len(pred[0][0])):
                for l in range(len(pred[0][0][0])):
                    sum += (pred[i][j][k][l] - yte[i][j][k][l]) ** 2
    rmse_ha = (sum / N) ** 0.5
    rmse_ha = cal_real_rmse(rmse_ha)
    # rmse_ha = rmse(yte, pred)
    print('window_len = {}, the RMSE of HA model is {}'.format(window_len, rmse_ha))
    return rmse_ha


########################################
# test for HA model
# window_lens = [i for i in range(1, 50)]
# for window_len in window_lens:
#     xtr, ytr, xte, yte = load_BikeNYC(window_len=window_len, nb_flow=2, len_test=240)
#     rmse_ha = HA_test(xte=xte, yte=yte, window_len=window_len)


def build_model(window_len, lr):
    model = myCrossLayer(nb_flow=2, map_height=16, map_width=8, nb_layers=3, window_len=window_len, nb_filter=1)
    my_optim = adam(lr=lr)
    model.compile(loss='mse', optimizer=my_optim, metrics=[rmse])
    model.summary()
    plot_model(model, to_file='my_hybrid_model.png', show_shapes=True)
    return model


model_name = './others/model_best.h5'

window_len = 15
lr = 0.001
xtr, ytr, xte, yte = load_BikeNYC_new(window_len=window_len, nb_flow=2, len_test=240)
epochs = 100
batch_size = 64
model = build_model(window_len=window_len, lr=lr)
print('hybrid model built ... ')
earlystoping = EarlyStopping(monitor='val_root_mean_square_error', patience=10, mode='min')
checkpoint = ModelCheckpoint(filepath=model_name, monitor='val_root_mean_square_error', verbose=1, save_best_only=True,mode='min', period=1)
checkpoint2 = ModelCheckpoint(filepath='./others/model_best_2.h5', monitor='val_root_mean_square_error', verbose=1, save_best_only=True,mode='min', period=1)

hist = model.fit(x=xtr, y=ytr, batch_size=batch_size, epochs=epochs, validation_split=0.1,shuffle=True, callbacks=[earlystoping,checkpoint])

print('the first time to evaluate test set')
score = model.evaluate(x=xte, y=yte, batch_size=32)
print('raw rmse is', score[1])
print('the real RMSE norm_01 is ', cal_real_rmse_01norm(np.sqrt(score[0])))
# print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))
print('##'*30)

print('begin to use the best model on validation set to evaluate the test set')
model.load_weights(model_name)

score = model.evaluate(x=xte, y=yte, batch_size=32)
print('raw rmse is', score[1])
print('the real RMSE norm_01 is ', cal_real_rmse_01norm(np.sqrt(score[0])))
# print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))


print('love world')
"""to fine tune the model"""
# print('$$'*30)
# print('begin to fine tune the model')
# hist_2 = model.fit(x=xtr, y=ytr, batch_size=batch_size, epochs=epochs, validation_split=0.1,shuffle=True, callbacks=[earlystoping,checkpoint2])
#
# print('the first time to evaluate test set')
# score = model.evaluate(x=xte, y=yte, batch_size=32)
# print('the real RMSE norm_01 is ', cal_real_rmse_01norm(np.sqrt(score[0])))
# print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))
# print('##'*30)
#
# print('begin to use the best model on validation set to evaluate the test set')
# model.load_weights(model_name)
#
# score = model.evaluate(x=xte, y=yte, batch_size=32)
# print('the real RMSE norm_01 is ', cal_real_rmse_01norm(np.sqrt(score[0])))
# print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))




