import keras
from keras.backend import squeeze, expand_dims

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.concatenate([x1, x2], axis=1)
added_new = expand_dims(added, axis=-1)

# out = keras.layers.Dense(4)(added_new)
# model = keras.models.Model(inputs=[input1, input2], outputs=out)
model = keras.models.Model(inputs=[input1, input2], outputs=added_new)

model.summary()

print('hello world')
