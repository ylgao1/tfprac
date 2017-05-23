import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from sklearn.metrics import r2_score

learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10
TRAINING_STEPS = 3000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y):
    func_lstm_cell = lambda : tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)
    cell = tf.contrib.rnn.MultiRNNCell([func_lstm_cell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    print(output.get_shape())
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    pred = tf.contrib.layers.fully_connected(output, 1, None)
    labels = tf.reshape(y, [-1])
    pred = tf.reshape(pred, [-1])

    loss = tf.losses.mean_squared_error(pred, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                               optimizer='Adagrad', learning_rate=0.1)
    return pred, loss, train_op


regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir='temp/c8/mdl2'))

test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP

train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES, dtype=np.float32
)))

test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES, dtype=np.float32
)))

regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

pred_y = regressor.predict(test_X)

r2 = r2_score(test_y, pred_y)

print(f'R2 score: {r2}')

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
#
# ax1.plot(pred_y, label='pred')
# ax2.plot(test_y, label='real_sin')
#
# fig.savefig('temp/c8/sin.png')















