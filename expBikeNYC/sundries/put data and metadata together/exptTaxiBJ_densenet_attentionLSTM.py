# from utils.evaluation_metrics import rmse
from models.CrossLayer_densenet_attentionLSTM import myCrossLayer
from utils.eval_metric import rmse
from sundries.TaxiBJ import load_data
import numpy as np
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
start_time = time.time()


def cal_real_rmse(x):
    m_factor = (16 * 8 / 81) ** 0.5
    return x / 2 * 267 * m_factor


def cal_real_rmse_01norm(x):
    m_factor = (16 * 8 / 81) ** 0.5
    return x * 267 * m_factor


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


def plot_real_rmse_train_val():
    import matplotlib.pyplot as plt
    train = hist.history['root_mean_square_error']
    val = hist.history['val_root_mean_square_error']
    train = [cal_real_rmse_01norm(i) for i in train]
    val = [cal_real_rmse_01norm(j) for j in val]
    test = cal_real_rmse_01norm(score[1])
    epo = [i for i in range(len(train))]
    plt.xlabel('epoch')
    plt.ylabel('RMSE_loss')
    plt.plot(epo, train, color='green')
    plt.plot(epo, val, color='red')
    plt.scatter(100, test, marker='v', c='black')
    plt.legend(['train', 'validation'])
    plt.show()

model_name = './others/{}_best.h5'.format(time.strftime("%Y-%m-%d %X", time.localtime()))

window_len = 6
nb_filter = 4
filter_size = 3
nb_layer = 3
lr = 0.001
xtr, ytr, xte, yte, external_dim = load_data(window_len=window_len, nb_flow=2, len_test=1304)
epochs = 500
batch_size = 64
# model = build_model(window_len=window_len, lr=lr)
model = myCrossLayer(nb_flow=2, map_height=16, map_width=8, external_dim=external_dim,
                     nb_layers=nb_layer, window_len=window_len, nb_filter=nb_filter, filter_size=filter_size)
my_optim = adam(lr=lr)
model.compile(loss='mse', optimizer=my_optim, metrics=[rmse])
model.summary()
plot_model(model, to_file='dense_model.png', show_shapes=True)
print('hybrid model built ... ')

# earlystoping = EarlyStopping(monitor='val_root_mean_square_error', patience=30, mode='min')
earlystoping = EarlyStopping(monitor='val_loss', patience=30, mode='min')
checkpoint = ModelCheckpoint(filepath=model_name, monitor='val_root_mean_square_error',
                             verbose=1, save_best_only=True, mode='min',
                             period=1)  # checkpoint2 = ModelCheckpoint(filepath='./others/model_best_2.h5', monitor='val_root_mean_square_error', verbose=1, save_best_only=True,mode='min', period=1)
hist = model.fit(x=xtr, y=ytr, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
                 callbacks=[earlystoping, checkpoint])

print('the first time to evaluate test set')
score = model.evaluate(x=xte, y=yte, batch_size=32)
print('raw rmse is', score[1])
print('the real RMSE norm_01 is ', cal_real_rmse_01norm(np.sqrt(score[0])))
# print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))
print('##' * 30)

print('begin to use the best model on validation set to evaluate the test set')
model.load_weights(model_name)

score = model.evaluate(x=xte, y=yte, batch_size=32)
print('raw rmse is', score[1])
print('the real RMSE norm_01 is ', cal_real_rmse_01norm(np.sqrt(score[0])))
# print('the real RMSE is ', cal_real_rmse(np.sqrt(score[0])))

end_time = time.time()
del_time = end_time - start_time
print('the whole process use {} minutes'.format(del_time / 60))

plot_train_val()
print('love world')
"""to fine tune the model"""
# print('$$'*30)
# print('begin to fine tune the model')
# hist_2 = model.fit(x=xtr, y=ytr, batch_size=batch_size, epochs=epochs, validation_split=0.1,shuffle=True, callbacks=[earlystoping,checkpoint2])

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
