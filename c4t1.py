import tensorflow as tf


w = tf.constant([[1.0,-2.0],[-3.0,4.0]])

l2 = tf.contrib.layers.l2_regularizer(1.0)(w)
l1 = tf.contrib.layers.l1_regularizer(1.0)(w)

sess = tf.InteractiveSession()

