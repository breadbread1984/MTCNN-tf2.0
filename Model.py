#!/usr/bin/python3

import tensorflow as tf;

def PNet():

  inputs = tf.keras.Input((None, None, 3,));
  results = tf.keras.layers.Conv2D(filters = 10, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(inputs);
  results = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  results = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  probs = tf.keras.layers.Conv2D(filters = 2, kernel_size = (1,1), padding = 'same', activation = tf.keras.layers.Softmax())(results);
  outputs = tf.keras.layers.Conv2D(filters = 4, kernel_size = (1,1), padding = 'same')(results);
  return tf.keras.Model(inputs = inputs, outputs = (probs, outputs));

def RNet():

  inputs = tf.keras.Input((24, 24, 3,));
  results = tf.keras.layers.Conv2D(filters = 28, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(inputs);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.Conv2D(filters = 48, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid')(results);
  results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (2,2), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  results = tf.keras.layers.Dense(units = 128, activation = tf.keras.layers.LeakyReLU())(results);
  probs = tf.keras.layers.Dense(units = 2, activation = tf.keras.layers.Softmax())(results);
  outputs = tf.keras.layers.Dense(units = 4)(results);
  return tf.keras.Model(inputs = inputs, outputs = (probs, outputs));

def ONet():

  inputs = tf.keras.Input((48, 48, 3,));
  results = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(inputs);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid')(results);
  results = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'same')(results);
  results = tf.keras.layers.Conv2D(filters = 128, kernel_size = (2,2), padding = 'valid', activation = tf.keras.layers.LeakyReLU())(results);
  results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.LeakyReLU())(results);
  probs = tf.keras.layers.Dense(units = 2, activation = tf.keras.layers.Softmax())(results);
  outputs1 = tf.keras.layers.Dense(units = 4)(results);
  outputs2 = tf.keras.layers.Dense(units = 10)(results);
  return tf.keras.Model(inputs = inputs, outputs = (probs, outputs1, outputs2));

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  pnet = PNet();
  rnet = RNet();
  onet = ONet();

