from keras import Model, Input
from keras.layers import Dense, Conv2D, Flatten, Reshape, LSTM
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.optimizers import adam
from keras.losses import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.iLayer import iLayer

def basic_conv2d(nb_filter, nb_row, nb_col, padding='same', bn=False):
    """
    basic conv2d layers
    :param nb_filter:
    :param nb_row:
    :param nb_col:
    :param padding:
    :param bn:
    :return:
    """

    def f1(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Conv2D(filters=nb_filter, kernel_size=(nb_col, nb_row), padding=padding)(activation)

    return f1


def dense_conv3D(nb_filter=64, nb_row=3, nb_col=3, padding='same', bn=False, nb_layers=3):
    """
    to stack a series of cnn layers and concatenate them with different weights
    :param nb_filter:   number of filters default 64
    :param nb_row:  kernel size
    :param nb_col:  kernel size
    :param padding:  padding methods in cnn, default 'same'
    :param bn:   BatchNormalization choice, boolean, default False
    :param nb_layers: number of cnn layers to be stacked
    :return:
    """

    def f2(input):
        cnn_fea = []
        tmp = input
        for i in range(nb_layers):
            tmp = basic_conv2d(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, padding=padding, bn=bn)(tmp)
            cnn_fea.append(tmp)
        # allocate weights to 3 cnn features
        new_features = []
        for fea in cnn_fea:
            new_features.append(iLayer()(fea))
        # conatenate the 3 cnn layer features
        res = Concatenate(axis=1)(new_features)
        return res

    return f2


def add_lstm(nb_flow=2, map_height=16, map_width=8):
    """
    use lstm to extract the temporal relation hidden in the features extracted from cnn
    :param nb_flow:
    :param map_height:
    :param map_width:
    :return:
    """

    def f3(data_in):
        main_output1 = Reshape((64 * 3 * map_height, map_width))(data_in)
        main_output2 = LSTM(units=nb_flow * map_height * map_width)(main_output1)
        main_output2 = Activation('tanh')(main_output2)
        data_out = Reshape((nb_flow, map_height, map_width))(main_output2)
        return data_out

    return f3


def myCrossLayer(nb_flow, map_height=16, map_width=8, nb_layers=3):
    """
    the final model
    :param nb_flow: number of measurements, also number of channels of each picture sample
    :param map_height: grid map height, here is 16
    :param map_width: grid map width, here is 8
    :param nb_layers: number of cnn layers
    :return:
    """
    nb_flow = nb_flow
    inputs = Input(shape=(nb_flow, map_height, map_width))
    main_inputs = []
    for i in range(6):
        main_inputs.append(inputs)
    cnn_fea = dense_conv3D(nb_filter=64, nb_col=3, nb_row=3, padding='same', nb_layers=nb_layers)(inputs)
    outputs = add_lstm(nb_flow=nb_flow, map_height=map_height, map_width=map_width)(cnn_fea)
    model = Model(input=main_inputs, output=outputs)
    return model


def build_model(lr=0.001):
    model = myCrossLayer(nb_flow=2, map_height=16, map_width=8, nb_layers=3)
    my_optim = adam(lr=lr)
    model.compile(loss='mse', optimizer=my_optim, metrics=[mean_squared_error])
    model.summary()
    plot_model(model, to_file='my_hybrid_model.png', show_shapes=True)

if __name__ == '__main__':
    model = build_model(lr=0.001)

    print('love world')
