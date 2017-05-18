import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets('temp/MNIST_data', one_hot=True, dtype=tf.uint8)

images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = 'temp/c7/c7.tfrec'

writer = tf.python_io.TFRecordWriter(filename)
for idx in range(num_examples):
    image_raw = images[idx].tostring()