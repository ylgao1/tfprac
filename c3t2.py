import tensorflow as tf



a = tf.constant([1,2], name='a')
b = tf.constant([2,3], name='b')
c = tf.add(a, b, name='add')

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config=config)

