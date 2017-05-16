import tensorflow as tf

# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
#
# saver = tf.train.Saver()
# sess = tf.InteractiveSession()
# saver.restore(sess, 'temp/c5.ckpt')
# print(sess.run(result))


# saver = tf.train.import_meta_graph('temp/c5.ckpt.meta')
# sess = tf.InteractiveSession()
# saver.restore(sess, 'temp/c5.ckpt')
# g = tf.get_default_graph()
# print(sess.run(g.get_tensor_by_name('add:0')))


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
result = v1 + v2

saver = tf.train.Saver({'v1':v1, 'v2':v2})
sess = tf.InteractiveSession()
saver.restore(sess, 'temp/c5/c5.ckpt')
print(sess.run(result))
















