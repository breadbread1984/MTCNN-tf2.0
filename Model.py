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
  results = tf.keras.layers.Flatten()(results);
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
  results = tf.keras.layers.Flatten()(results);
  results = tf.keras.layers.Dense(units = 256, activation = tf.keras.layers.LeakyReLU())(results);
  probs = tf.keras.layers.Dense(units = 2, activation = tf.keras.layers.Softmax())(results);
  outputs1 = tf.keras.layers.Dense(units = 4)(results);
  outputs2 = tf.keras.layers.Dense(units = 10)(results);
  return tf.keras.Model(inputs = inputs, outputs = (probs, outputs1, outputs2));

class MTCNN(tf.keras.Model):

  def __init__(self, minsize = 20, threshold = [0.6, 0.7, 0.7], factor = 0.709):

    super(MTCNN, self).__init__();
    # minsize: minimum face size.
    # threshold: (thr1, thr2, thr3) are thresholds for pnet, rnet and onet.
    # factor: factor for image pyramid.
    assert type(minsize) is int;
    assert type(threshold) is list and len(threshold) == 3;
    assert type(factor)is float;
    self.pnet = PNet();
    self.rnet = RNet();
    self.onet = ONet();
    self.minsize = minsize;
    self.threshold = threshold;
    self.factor = factor;

  def getBBox(self, outputs, probs, scale):

    dx1,dy1,dx2,dy2 = tf.unstack(outputs, axis = -1);
    pos = tf.where(tf.math.greater_equal(outputs, self.threshold[0]));
    # TODO:

  def call(self, inputs):

    imgs = tf.cast(inputs, dtype = tf.float32);
    # generate image pyramid scales
    m = (12.0 / self.minsize) * tf.math.reduce_min(inputs.shape[1:3]);
    scale = (12.0 / self.minsize);
    scales = list();
    while m >= 12:
      scales.append(scale);
      scale = scale * self.factor;
      m = m * self.factor;
    # first stage
    for scale in scales:
      sz = tf.math.ceil(inputs.shape[1:3] * scale);
      imgs = (tf.image.resize(inputs, sz) - 127.5) / 128.0;
      probs, outputs = self.pnet(imgs);
      bbox = self.getBBox(outputs, probs, scale);

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  pnet = PNet();
  rnet = RNet();
  onet = ONet();
  print(pnet.outputs[0].shape);
  print(rnet.outputs[0].shape);
  print(onet.outputs[0].shape);
