import tensorflow as tf

# sess.close()

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1), name='w1')
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1), name='w2')
#w3 = tf.Variable(w1.initialized_value(), name='w3', trainable=False)

#x = tf.constant([0.7,0.9], shape=[1,2])
x = tf.placeholder(tf.float32, shape=[None,2], name='input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.InteractiveSession()

init_op = tf.global_variables_initializer()
sess.run(init_op)

xs = [
    [0.7,0.9],
    [0.1,0.4],
    [0.5,0.8]
]

yp = y.eval(feed_dict={x:xs})

# v1 = tf.global_variables()
# v2 = tf.trainable_variables()