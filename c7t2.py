import tensorflow as tf

reader = tf.TFRecordReader()

fn_q = tf.train.string_input_producer(['temp/c7/c7.tfrec'])
_, sexp = reader.read(fn_q)

features = tf.parse_single_example(sexp, features={
    'image_raw': tf.FixedLenFeature([], tf.string),
    'pixels': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64)
})

images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.InteractiveSession()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

image, label, pixel = sess.run([images, labels, pixels])























