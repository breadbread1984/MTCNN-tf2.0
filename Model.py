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
      # boxes.shape = (target num, 9)
      boxes = tf.concat([upper_left, down_right, tf.expand_dims(score, -1), reg], axis = -1);
      boxes_batch.append(boxes);
    # boxes_batch.shape = batch * (target num, 4 (anchorbox pos) + 1 (objectness) + 4 (regressed deviation))
    return boxes_batch;

  def nms(self, boxes_batch, threshold = 0.5, method = 'union'):

    indices_batch = list();
    for boxes in boxes_batch:
      down_right = boxes[...,2:4];
      upper_left = boxes[...,0:2];
      # hw.shape = (target num, 2)
      hw = down_right - upper_left + tf.ones((1,2,));
      # area.shape = (target num,)
      area = hw[...,0] * hw[...,1];
      # sort with respect to weight in descend order.
      descend_idx = tf.argsort(boxes[...,4], direction = 'DESCENDING');
      i = 0;
      while i < descend_idx.shape[0]:
        # idx = ()
        idx = descend_idx[i];
        # cur_xxx.shape = (1, 2)
        cur_upper_left = upper_left[idx:idx + 1, ...];
        cur_down_right = down_right[idx:idx + 1, ...];
        # area.shape = (1,)
        hw = cur_down_right - cur_upper_left + tf.ones((1, 2), dtype = tf.float32);
        area = hw[0] * hw[1];
        # following_idx.shape = (following number,)
        following_idx = descend_idx[i + 1:];
        # following_xxx.shape = (following number, 2)
        following_upper_left = tf.gather_nd(upper_left, following_idx);
        following_down_right = tf.gather_nd(down_right, following_idx);
        # following_area.shape = (following number,)
        following_hw = following_down_right - following_upper_left + tf.ones((1,2), dtype = tf.float32);
        following_area = following_hw[0] * following_hw[1];
        # intersect_hw.shape = (following number, 2)
        # negative means no intersection.
        max_upper_left = tf.math.maximum(cur_upper_left, following_upper_left);
        min_down_right = tf.math.minimum(cur_upper_left, following_down_right);
        intersect_hw = min_down_right - max_upper_left + tf.ones((1, 2), dtype = tf.float32);
        # intersect_area.shape = (following number)
        intersect_hw = tf.where(tf.math.greater(intersect_hw, 0), intersect_hw, tf.zeros_like(intersect_hw));
        intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1];
        # overlap rate
        if method.lower() is 'min':
          overlap = intersect_area / tf.math.minimum(area, following_area);
        elif method.lower() is 'union':
          overlap = intersect_area / (area + following_area - intersect_area);
        else: raise Exception('unknown overlap method!');
        # pick following idx with overlap lower than given one.
        indices = tf.where(tf.less(overlap, threshold));
        following_idx = tf.gather(following_idx, indices);
        descend_idx = tf.concat([descend_idx[:i], following_idx], axis = -1);
        i += 1;
      indices_batch.append(descend_idx);
    return indices_batch;

  def toSquare(self, total_boxes):

    # hw.shape = (target num, 2)
    hw = total_boxes[..., 2:4] - total_boxes[..., 0:2];
    # length.shape = (target num)
    length = tf.math.reduce_max(hw, axis = -1);
    # center.shape = (target num, 2)
    center = total_boxes[..., 0:2] + hw * 0.5;
    # shape = (target num, 2)
    upper_left = center - tf.stack([length, length], axis = -1) * 0.5;
    down_right = upper_left + tf.stack([length, length], axis = -1);
    total_boxes = tf.concat([upper_left, down_right, total_boxes[...,4:5]], axis = -1);
    return total_boxes;

  def clip(self, total_boxes, input_shape):

    boxes = total_boxes[...,0:4];
    lower = tf.constant([1, 1, 1, 1], dtype = tf.float32);
    upper = tf.cast(tf.concat([input_shape[1:3], input_shape[1:3]], axis = -1), dtype = tf.float32);
    boxes = tf.clip_by_value(boxes,lower,upper);
    total_boxes = tf.concat([boxes, total_boxes[...,4:5]], axis = -1);
    return total_boxes;

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
    # 1)first stage
    # nms among targets of the same scale.
    total_boxes = [tf.zeros((0,9), dtype = tf.float32) for i in tf.range(imgs.shape[0])];
    for scale in scales:
      sz = tf.math.ceil(inputs.shape[1:3] * scale);
      imgs = (tf.image.resize(inputs, sz) - 127.5) / 128.0;
      probs, outputs = self.pnet(imgs);
      # channel-1 of probs represents is a face.
      boxes_batch = self.getBBox(outputs, probs[...,1], scale);
      indices_batch = self.nms(boxes_batch, 0.5, 'union');
      for b in tf.range(boxes_batch.shape[0]):
        boxes = boxes_batch[b];
        indices = indices_batch[b];
        total_boxes[b] = tf.concat([total_boxes[b], tf.gather(boxes, indices)], axis = 0);
    # nms among targets of different scales.
    indices_batch = self.nms(total_boxes, 0.7, 'union');
    for b in tf.range(len(total_boxes)):
      boxes = total_boxes[b];
      indices = indices_batch[b];
      boxes = tf.gather_nd(boxes, indices);
      # hw.shape = (target number, 2)
      hw = boxes[..., 2:4] - boxes[..., 0:2];
      # bounding.shape = (target number, 4 (bounding) + 1 (objectness))
      bounding = boxes[...., 0:4] + tf.concat([hw[..., 0:2],hw[..., 0:2]],axis = -1) * boxes[..., 5:9];
      bounding = tf.concat([bounding, boxes[..., 4]], axis = -1);
      # from rectangle bounding to square bounding
      bounding = self.toSquare(bounding);
      # clip the bounding within image
      bounding = self.clip(bounding, inputs.shape);
      # total_boxes.shape = batch * (target number, 5)
      total_boxes[b] = bounding;
    # second stage
    for b in tf.range(len(total_boxes)):
      # boxes.shape = (target num, 5)
      boxes = total_boxes[b];
      img = inputs[b:b+1,...];
      # crop target and resize
      target_imgs = tf.image.crop_and_resize(img, boxes[...,0:4], tf.zeros((boxes.shape[0]), dtype = tf.int32), (24,24));
      target_imgs = (target_imgs - 127.5) / 128;
      probs, outputs = self.rnet(target_imgs);
      valid_indices = tf.where(tf.math.greater(probs[...,1], self.threshold[1]));
      boxes = tf.gather_nd(boxes, valid_indices);
      scores = tf.gather_nd(probs[...,1], valid_indices);
      total_boxes[b] = tf.concat([boxes[..., 0:4], scores], axis = -1);
    indices_batch = self.nms(total_boxes, 0.7, 'union');
    for b in tf.range(len(total_boxes)):
      boxes = total_boxes[b];
      indices = indices_batch[b];
      boxes = tf.gather_nd(boxes, indices);

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  pnet = PNet();
  rnet = RNet();
  onet = ONet();
  print(pnet.outputs[0].shape);
  print(rnet.outputs[0].shape);
  print(onet.outputs[0].shape);
