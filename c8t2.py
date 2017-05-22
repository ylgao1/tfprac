import tensorflow as tf
import c8_reader as reader

DATA_PATH = 'temp/ptb/simple-examples/data'
train_data, valid_data, test_data, vocablary = reader.ptb_raw_data(DATA_PATH)

result = reader.ptb_producer(train_data, 4, 5)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for i in range(3):
        x, y = sess.run(result)
        print(f'X{i}: {x}')
        print(f'Y{i}: {y}')
    coord.request_stop()
    coord.join(threads)



