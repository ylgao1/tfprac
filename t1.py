import tensorflow as tf

a = 10
i = tf.train.range_input_producer(a, shuffle=True).dequeue()

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)




# coord.request_stop()
# coord.join(threads)