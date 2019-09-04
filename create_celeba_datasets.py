#!/usr/bin/python3

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
    Num = 0;
    Head = 1;
    Anno = 2;
  # context info
  count = 0;
  s = Status.Num;
  for line in f:
    line = line.strip();
    if line == '': break;
    if s == Status.Num:
      count = int(line);
      s = Status.Head;
    elif s == Status.Head:
      # do nothing, just consume the line.
      s = Status.Anno;
    elif s == Status.Anno:
      tokens = line.split();
      assert len(tokens) == 11;
      img = cv2.imread(join(root_dir, tokens[0]));
      assert img is not None;
      landmarks = np.array([[int(tokens[1]), int(tokens[2])],
                            [int(tokens[3]), int(tokens[4])],
                            [int(tokens[5]), int(tokens[6])],
                            [int(tokens[7]), int(tokens[8])],
                            [int(tokens[9]), int(tokens[10])]], dtype = np.int32)
      # landmarks = np.flip(landmarks, axis = 1);
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img.tobytes()])),
          'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = list(img.shape))),
          'landmarks': tf.train.Feature(int64_list = tf.train.Int64List(value = landmarks.reshape(-1)))
        }
      ));
      writer.write(trainsample.SerializeToString());
      count -= 1;
      s = Status.Anno;
  assert count == 0;
  f.close();
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  main('img_celeba', 'list_landmarks_celeba.txt', 'celeba_train.tfrecord');
