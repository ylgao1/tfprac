import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
lr = 0.001
dropout = 0.9

IMAGE_PIXEL = 784
IMAGE_SIZE = 28
NUM_CLASSES = 10

data_dir = 'temp/MNIST_data'
log_dir = 'temp/tb'

if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
else:
    tf.gfile.MakeDirs(log_dir)

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y-input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    tf.summary.image('input', image_shaped_input, 10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


hidden1 = nn_layer(x, IMAGE_PIXEL, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, NUM_CLASSES, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir+'/test')
tf.global_variables_initializer().run()


def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x:xs, y_:ys, keep_prob:k}


saver = tf.train.Saver()
for i in range(max_steps):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print(f'Accuracy at step {i}:{acc}')
    else:
        if i % 100 == 99:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+'/model.ckpt', i)
            print(f'Adding run metadata for {i}')
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()


























