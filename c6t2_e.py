import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import c6t2 as mn


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mn.IMAGE_SIZE, mn.IMAGE_SIZE, mn.NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mn.OUTPUT_NODE], name='y-input')

        validate_feed = {x: mnist.validation.images.reshape(-1, mn.IMAGE_SIZE, mn.IMAGE_SIZE, mn.NUM_CHANNELS),
                         y_: mnist.validation.labels}
        y = mn.inference(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mn.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print(f'After {global_step} training step(s), validation accuracy = {accuracy_score:g}')
            else:
                print('No checkpoint file found')
                return


def main(argv=None):
    mnist = input_data.read_data_sets('temp/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
