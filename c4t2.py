import tensorflow as tf
import numpy as np


def get_weight(shape, lmbd):
    var = tf.Variable(tf.random_normal(shape))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lmbd)(var))
    return var


x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

batch_size = 8

layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

y = tf.sigmoid(cur_layer)
xentropy = -tf.reduce_mean(y_ * tf.log(y))
tf.add_to_collection('losses', xentropy)
loss = tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

np.random.seed(1)
dataset_size = 128
X = np.random.rand(dataset_size, 2)
Y = np.array([int(x1 + x2 < 1) for (x1, x2) in X], dtype=np.float32)[None].T

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_op, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
            print(f'After {i} trainning step(s), cross entropy on all data is {total_cross_entropy:g}')
