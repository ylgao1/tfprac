import tensorflow as tf

filename = 'temp/c7/c7.tfrec'

fn_q = tf.train.string_input_producer([filename], num_epochs=1)
reader = tf.TFRecordReader()

_, sdata = reader.read(fn_q)
features = tf.parse_single_example(sdata, features={
    'pixels': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string)
})

image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

image_p = tf.reshape(image, (784,))

batch_size = 32
qcapacity = 1000 + 3 * batch_size

image_b, label_b= tf.train.shuffle_batch([image_p, label], batch_size=batch_size, capacity=qcapacity,
                                num_threads=3, min_after_dequeue=30)


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)






















