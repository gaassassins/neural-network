# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist
import numpy as np
from PIL import Image
from keras import backend as K


num_classes = 10
im = Image.open("2.png")
img_rows, img_cols = 28, 28
im_grey = im.convert('L')
im_array = np.array(im_grey)
im_array = np.reshape(im_array, (1, img_rows, img_cols, 1)).astype('float32')
x = 255 - im_array
x /= 255

print("Загружаю сеть из файлов")
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('mnist_model.h5')
print("Загрузка сети завершена")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

scores = loaded_model.evaluate(x_test, y_test, verbose=0)
print("Точность работы загруженной сети на тестовых данных: %.3f%%" % (scores[1]))
prediction = loaded_model.predict(x)
prediction = np_utils.categorical_probas_to_classes(prediction)
print(prediction)