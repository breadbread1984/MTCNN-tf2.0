#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def main():

  # load dataset
  celeba_builder = tfds.builder("celeb_a");
  celeba_builder.download_and_prepare();
  # try to load the dataset once
  celeba_train = tfds.load(name = "celeb_a", split = tfds.Split.TRAIN, download = False);
  celeba_test = tfds.load(name = "celeb_a", split = tfds.Split.TEST, download = False);

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  main();
