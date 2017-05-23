import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10

TRAINING_STEPS = 10000
BATCH_SIZE = 50

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.concatenate(X).astype(np.float32).reshape((-1, 1, TIMESTEPS)), np.array(y, dtype=np.float32)


test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP

train_x, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES, dtype=np.float32
)))

test_x, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES, dtype=np.float32
)))

x = tf.placeholder(tf.float32, [None, None, TIMESTEPS])
y_ = tf.placeholder(tf.float32, [None, 1])

lstm_cell = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
output = tf.reshape(output, [-1, HIDDEN_SIZE])
weight = tf.get_variable('weight', [HIDDEN_SIZE, 1])
bias = tf.get_variable('bias', [1])
pred = tf.matmul(output, weight) + bias
loss = tf.losses.mean_squared_error(y_, pred)
train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)







xtr = tf.convert_to_tensor(train_x, tf.float32)
ytr = tf.convert_to_tensor(train_y, tf.float32)
qcapacity = 1000 + 3 * BATCH_SIZE
xtrb, ytrb = tf.train.shuffle_batch([xtr, ytr], batch_size=BATCH_SIZE, enqueue_many=True,
                                capacity=qcapacity, min_after_dequeue=100)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

for step in range(TRAINING_STEPS):
    xs, ys = sess.run([xtrb, ytrb])
    _, loss_value= sess.run([train_op, loss], feed_dict={x:xs, y_:ys})
    if step % 100 == 0:
        print(f'iteration {step}, loss value: {loss_value:g}')

ypred = sess.run(pred, feed_dict={x: test_x})
r2 = r2_score(test_y, ypred)

# xt = tf.convert_to_tensor(test_x, tf.float32)