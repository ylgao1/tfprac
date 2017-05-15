import tensorflow as tf


def foo():
    with tf.variable_scope('abc'):
        a = tf.get_variable('a', [1], tf.float32, initializer=tf.ones_initializer)
    return a


def foo1():
    a = foo()
    b = a + 1
    return b


b = foo1()

with tf.variable_scope('abc', reuse=True):
    a1 = tf.get_variable('a')


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()