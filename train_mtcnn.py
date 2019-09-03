#!/usr/bin/python3

import cv2;
import tensorflow as tf;
from Model import MTCNN;

def parse_widerface_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string),
      'shape': tf.io.FixedLenFeature((3,), dtype = tf.int64),
      'objects': tf.io.VarLenFeature(dtype = tf.int64),
      'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
    }
  );
  shape = tf.cast(feature['shape'], dtype = tf.int32);
  data = tf.io.decode_raw(feature['data'], out_type = tf.uint8);
  data = tf.reshape(data, shape);
  obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
  objects = tf.sparse.to_dense(feature['objects'], default_value = 0);
  objects = tf.reshape(objects, (obj_num, 10));
  return data, objects;

def main():

  widerface_trainset = tf.data.TFRecordDataset('wider_face_train.tfrecord').map(parse_widerface_function).batch(1);
  widerface_evalset = tf.data.TFRecordDataset('wider_face_val.tfrecord').map(parse_widerface_function).batch(1);
  for data, objects in widerface_trainset:

    cv2.imshow('debug',data[0].numpy());
    cv2.waitKey();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
