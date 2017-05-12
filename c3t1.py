import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
    with g1.device('/cpu:0'):
        a = tf.constant([1, 2, 3], shape=[3], name='a')
        b = tf.constant([1, 2, 3], shape=[3], name='b')
        c = a + b

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer)

with tf.Session(graph=g1, config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    print(sess.run(c))

with tf.Session(graph=g2, config=tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))
