import numpy as np
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1), name='w2')

x = tf.placeholder(tf.float32, shape=[None,2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

a = tf.matmul(x, w1)
ypa = tf.matmul(a, w2)
y = tf.sigmoid(ypa)

cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)



np.random.seed(1)
dataset_size = 128
X = np.random.rand(dataset_size, 2)
Y = np.array([int(x1+x2<1) for (x1,x2) in X], dtype=np.float32)[None].T

# with tf.Session()






















