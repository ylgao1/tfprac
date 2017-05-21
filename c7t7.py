import tensorflow as tf
import os

files = list(map(lambda c: f'temp/c7a/{c}', filter(lambda d: d.startswith('data'), os.listdir('temp/c7a'))))

fn_q = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, sexp = reader.read(fn_q)
features = tf.parse_single_example(sexp, features={
    'i': tf.FixedLenFeature([], tf.int64),
    'j': tf.FixedLenFeature([], tf.int64)
})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(8):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)


