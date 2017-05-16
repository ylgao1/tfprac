import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0))
        if avg_class is None:
            layer1 = tf.matmul(input_tensor, weights) + biases
        else:
            layer1 = tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases)
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0))
        if avg_class is None:
            layer2 = tf.matmul(layer1, weights) + biases
        else:
            layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)
    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    y = inference(x, None, reuse=False)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    print(tf.trainable_variables())
    average_y = inference(x, variable_averages, True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    with tf.variable_scope('layer1', reuse=True):
        weights1 = tf.get_variable('weights')
    with tf.variable_scope('layer2', reuse=True):
        weights2 = tf.get_variable('weights')
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    lr = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                    mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op('train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(f'After {i} training step(s), validation accuracy using average model is {validate_acc:g}')
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(f'After {TRAINING_STEPS} training step(s), test accuracy using average model is {test_acc:g}')


def main(argv=None):
    mnist = input_data.read_data_sets('temp/MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()