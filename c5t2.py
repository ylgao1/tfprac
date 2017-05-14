import tensorflow as tf


with tf.name_scope('layer1'):
    v = tf.Variable(1.0, name='w')


with tf.variable_scope('layer1', reuse=True):
    v1 = tf.get_variable('w', shape=[1])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
