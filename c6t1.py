import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 32, 32, 3])

filter_weight = tf.get_variable('weights', [5, 5, 3, 16],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(x, filter_weight, [1,1,1,1], 'SAME')
bias = tf.nn.bias_add(conv, biases)

actived_conv = tf.nn.relu(bias)
pool = tf.nn.max_pool(actived_conv, [1,3,3,1], [1,2,2,1], 'SAME')

