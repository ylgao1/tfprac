import tensorflow as tf
import numpy as np

np.random.seed(1337)

# data IO
data = open('input.txt').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'data has {data_size} characters, {vocab_size} unique.')
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}


# hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1




























