import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf


# CIFAR-10 데이터 다운로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

print(y_train_one_hot.shape)