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

example, label = features['i'], features['j']

batch_size = 240

capacity = 1000 + 3 * batch_size

example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                            batch_size=batch_size,
                                            capacity=capacity, min_after_dequeue=30, num_threads=3)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch]
        )
        print(i)
        print(cur_example_batch)
        print(cur_label_batch)

    coord.request_stop()
    coord.join(threads)















