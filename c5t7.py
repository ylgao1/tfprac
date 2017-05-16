import tensorflow as tf
from tensorflow.python.platform import gfile


sess = tf.InteractiveSession()

with gfile.FastGFile('temp/c5/c5a_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

result = tf.import_graph_def(graph_def, return_elements=['add:0'])