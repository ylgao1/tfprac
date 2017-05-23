import tensorflow as tf
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

learn = tf.contrib.learn

def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    logits = tf.contrib.layers.fully_connected(features, 3, tf.nn.softmax)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                               optimizer='Adam', learning_rate=0.01)
    return tf.argmax(logits, 1), loss, train_op


iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

x_train, x_test = map(np.float32, [x_train, x_test])

classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir='temp/c8/mdl'))
classifier.fit(x_train, y_train, steps=800)
y_pred = classifier.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc}')


















