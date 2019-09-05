#!/usr/bin/python3

import sys;
import enum;
from os.path import join, isfile, isdir;
import numpy as np;
import cv2;
import tensorflow as tf;

def main(root_dir, input_file, output_file):

  if isdir(root_dir) == False:
    print("invalid root diretory!");
    exit(1);
  if isfile(input_file) == False:
    print("invalid input file!");
    exit(1);
  f = open(input_file, "r");
  if f is None:
    print("input file can't be opened!");
    exit(1);
  writer = tf.io.TFRecordWriter(output_file);
  if writer is None:
    print("invalid output file!");
    exit(1);
  class Status(enum.Enum):
    Path = 0;
    Num = 1;
    Anno = 2;
  # context variables
  s = Status.Path;
  img_path = str();
  object_num = 1;
  objs = list();
  for line in f:
    line = line.strip();
    if line == '': break;
    if s == Status.Path:
      img_path = line;
      s = Status.Num;
    elif s == Status.Num:
      object_num = int(line);
      objs = list();
      s = Status.Anno;
    elif s == Status.Anno:
      if object_num == 0:
        # image with no targets should write into dataset as well
        img = cv2.imread(join(root_dir, 'images', img_path));
        assert img is not None;
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img.tobytes()])),
            'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = list(img.shape))),
            'objects': tf.train.Feature(int64_list = tf.train.Int64List(value = [])),
            'obj_num': tf.train.Feature(int64_list = tf.train.Int64List(value = [0]))
          }
        ));
        writer.write(trainsample.SerializeToString());
        s = Status.Path;
      else:
        tokens = line.split(' ');
        assert len(tokens) == 10;
        objs.append(np.array(tokens).astype('int32'));
        object_num -= 1;
        if object_num == 0:
          # add one sample
          objs = np.array(objs); # (object num, 10)
          img = cv2.imread(join(root_dir, 'images', img_path));
          assert img is not None;
          trainsample = tf.train.Example(features = tf.train.Features(
            feature = {
              'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
              'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = list(img.shape))),
              'objects': tf.train.Feature(int64_list = tf.train.Int64List(value = objs.reshape(-1))),
              'obj_num': tf.train.Feature(int64_list = tf.train.Int64List(value = [objs.shape[0]]))
            }
          ));
          writer.write(trainsample.SerializeToString());
          s = Status.Path;
  f.close();
  writer.close();

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  main('WIDER_train', 'wider_face_split/wider_face_train_bbx_gt.txt', 'wider_face_train.tfrecord');
  main('WIDER_val', 'wider_face_split/wider_face_val_bbx_gt.txt', 'wider_face_val.tfrecord');
