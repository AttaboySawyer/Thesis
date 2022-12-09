import tensorflow as tf
import pickle
# from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('E:/Data/Base Network Saves/unaltered_1.h5')
# seg_model = tf.keras.models.load_model('E:\Data\Base Network Saves\model_test1.h5')
print(model.summary())
# for layer in model.layers:
#     layer.trainable = False

# # print(model.layers[-3].output)
# dense1 = tf.keras.layers.Dense(512, name='snn_dense_1')(model.layers[-3].output)
# output = tf.keras.layers.Dense(256, name='snn_dense_2')(dense1)

# anchor_in = tf.keras.Input(shape=(200,200,3))

# # model = model.layers[-3].output

# embedding = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
# print(embedding.summary())

# print(model.summary())

# with open('E:/Data/Base Network Saves/history saves/unaltered_1', "rb") as file_pi:
#     history = pickle.load(file_pi)
#     print(history)