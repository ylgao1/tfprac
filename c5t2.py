import tensorflow as tf


sess = tf.InteractiveSession()

def foo():
    with tf.variable_scope('abc'):
        with tf.variable_scope('bbc'):
            a = tf.get_variable('a', [1], tf.float32, initializer=tf.ones_initializer)
            print(tf.get_variable_scope().reuse)
    return a


def foo1():
    a = foo()
    b = a + 1
    return b


b = foo1()

with tf.variable_scope('abc', reuse=True):
    with tf.variable_scope('bbc'):
        a1 = tf.get_variable('a', [1], initializer=tf.constant_initializer(12.0))
        print(tf.get_variable_scope().reuse)




tf.global_variables_initializer().run()