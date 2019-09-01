#!/usr/bin/python3

import cv2;
import tensorflow as tf;
from Model import MTCNN;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string, default_value = ''),
      'shape': tf.io.FixedLenFeature((), dtype = tf.int64, default_value = 0),
      'objects': tf.io.FixedLenFeature((), dtype = tf.int64, default_value = 0),
      'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64, default_value = 0)
    }
  };
  shape = tf.reshape(feature['shape'],(2,));
  data = tf.io.decode_raw(feature['data'], out_type = tf.uint8);
  data = tf.reshape(data, shape + [3,]);
  obj_num = tf.reshape(feature['obj_num'], (1,));
  objects = tf.reshape(feature['objects'], obj_num + [10,]);
  return data, objects;

def main():

  widerface_trainset = tf.data.TFRecordDataset('wider_face_train.tfrecord').map(parse_function).batch(1);
  for data, objects in widerface_trainset:

    cv2.imshow('debug',data[0].numpy());
    cv2.waitKey();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
