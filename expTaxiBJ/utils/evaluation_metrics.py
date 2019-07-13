# from keras import losses as K
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np


# from keras import metrics
# epsilon = 1e-07


def mean_squared_error(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)


def root_mean_square_error(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100



# alias
mse = MSE = mean_squared_error
rmse = RMSE = root_mean_square_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error



# def confusion_matrix(y_true, y_pred):
#     confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
#     return confmat


