from keras import Model, Input
from keras.layers import Dense, Conv2D, Flatten, Reshape, LSTM, Lambda
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.optimizers import adam
from keras.backend import expand_dims
from keras.losses import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.iLayer import iLayer


def expand_dim_backend(x):
    """
    :param x:
    :return:
    """
    from keras import backend as K
    x1 = K.expand_dims(x, axis=1)
    return x1


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
            tmp = BatchNormalization()(tmp)
            cnn_fea.append(tmp)
        # allocate weights to 3 cnn features
        new_features = []
        for fea in cnn_fea:
            tmp1 = iLayer()(fea)
            tmp2 = Reshape(([nb_filter * 16 * 8]))(tmp1)
            tmp3 = Dense(units=1024, activation='tanh')(tmp2)
            new_features.append(tmp3)
            # cnn_fea_flatten = Reshape(([nb_layers * nb_filter * map_height * map_width]))(cnn_fea)

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
        # main_output1 = Reshape((64 * 3 *6* map_height, map_width))(data_in)
        main_output1 = Concatenate(axis=1)(data_in)
        main_output2 = LSTM(units=nb_flow * map_height * map_width)(main_output1)
        main_output2 = Activation('tanh')(main_output2)
        data_out = Reshape((nb_flow, map_height, map_width))(main_output2)
        return data_out

    return f3


def myCrossLayer(nb_flow=2, map_height=16, map_width=8, nb_layers=3, window_len=12, nb_filter=64):
    """
    the final model
    :param nb_flow: number of measurements, also number of channels of each picture sample
    :param map_height: grid map height, here is 16
    :param map_width: grid map width, here is 8
    :param nb_layers: number of cnn layers
    :return:
    """
    window_len_pic_fea = []
    main_inputs = []
    for i in range(window_len):
        inputs = Input(shape=(nb_flow, map_height, map_width))
        main_inputs.append(inputs)
        cnn_fea = dense_conv3D(nb_filter=nb_filter, nb_col=3, nb_row=3, padding='same', nb_layers=nb_layers)(inputs)
        cnn_fea_flatten = Reshape(([nb_layers * 1024]))(cnn_fea)
        # cnn_fea_flatten = expand_dims(cnn_fea_flatten, axis=1)
        cnn_fea_flatten = Lambda(expand_dim_backend)(cnn_fea_flatten)
        window_len_pic_fea.append(cnn_fea_flatten)

    outputs = add_lstm(nb_flow=nb_flow, map_height=map_height, map_width=map_width)(window_len_pic_fea)
    model = Model(inputs=main_inputs, outputs=outputs)
    return model


def build_model(lr=0.001):
    model = myCrossLayer(nb_flow=2, map_height=16, map_width=8, nb_layers=3, window_len=6)
    my_optim = adam(lr=lr)
    model.compile(loss='mse', optimizer=my_optim, metrics=[mean_squared_error])
    model.summary()
    plot_model(model, to_file='my_hybrid_model.png', show_shapes=True)


if __name__ == '__main__':
    model = build_model(lr=0.001)

    print('love world')
