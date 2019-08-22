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

  STRIDE = 2.0;
  CELLSIZE = 12.0;

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

    boxes_batch = list();
    reg_batch = list();
    for b in tf.range(output.shape[0]):
      # positions of targets over threshold
      # pos.shape = (target num, 2)
      pos = tf.where(tf.math.greater_equal(probs[b,...], self.threshold[0]));
      # pick the anchor box of targets over threshold
      # shape = (target num, 2)
      upper_left = tf.math.round((pos * self.STRIDE + 1) / scale);
      down_right = tf.math.round((pos * self.STRIDE + self.CELLSIZE - 1 + 1) / scale);
      # pick the score of tartgets over threshold
      # score.shape = (target num,)
      score = tf.gather_nd(probs[b,...], pos);
      # pick the regressed deviation of targets over threshold
      # reg.shape = (target num, 4)
      reg = tf.gather_nd(outputs[b,...], pos);
      # boxes.shape = (target num, 5)
      boxes = tf.concat([upper_left, down_right, tf.expand_dims(score, -1)], axis = -1);
      boxes_batch.append(boxes);
      reg_batch.append(reg);
    return boxes_batch, reg_batch;

  def nms(self, boxes_batch, threshold, method):

    for boxes in boxes_batch:
      down_right = boxes[...,2:4];
      upper_left = boxes[...,0:2];
      # hw.shape = (target num, 2)
      hw = down_right - upper_left + tf.ones((1,2,));
      # area.shape = (target num,)
      area = hw[...,0] * hw[...,1];
      # TODO

  def call(self, inputs):

    imgs = tf.cast(inputs, dtype = tf.float32);
    # generate image pyramid scales
    m = (self.CELLSIZE / self.minsize) * tf.math.reduce_min(inputs.shape[1:3]);
    scale = (self.CELLSIZE / self.minsize);
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
      # channel-1 of probs represents is a face.
      boxes_batch, _ = self.getBBox(outputs, probs[...,1], scale);


if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  pnet = PNet();
  rnet = RNet();
  onet = ONet();
  print(pnet.outputs[0].shape);
  print(rnet.outputs[0].shape);
  print(onet.outputs[0].shape);
