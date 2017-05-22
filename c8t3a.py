import numpy as np
import tensorflow as tf
import c8_reader as reader

DATA_PATH = 'temp/ptb/simple-examples/data'
HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1

log_dir = 'temp/c8/ld'
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
else:
    tf.gfile.MakeDirs(log_dir)

train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
train_data_len = len(train_data)
train_batch_len = train_data_len // TRAIN_BATCH_SIZE
train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP


train_q = reader.ptb_producer(train_data, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)


input_data = tf.placeholder(tf.int32, [TRAIN_BATCH_SIZE, TRAIN_NUM_STEP])
targets = tf.placeholder(tf.int32, [TRAIN_BATCH_SIZE, TRAIN_NUM_STEP])

lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(NUM_LAYERS)])

# cell = lstm_cell
init_state = cell.zero_state(TRAIN_BATCH_SIZE, tf.float32)

embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
inputs = tf.nn.embedding_lookup(embedding, input_data)

outputs = []
state = init_state
with tf.variable_scope('RNN'):
    for ts in range(TRAIN_NUM_STEP):
        if ts > 0:
            tf.get_variable_scope().reuse_variables()
        print(tf.get_variable_scope().name)
        lstm_output, state = cell(inputs[:,ts,:], state)
        outputs.append(lstm_output)

sess = tf.InteractiveSession()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
tf.global_variables_initializer().run()

x, y = sess.run(train_q)
fd = {input_data:x, targets:y}

train_writer.close()











