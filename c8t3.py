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

class PTBModel:
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    


































