from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from sklearn import cluster as c

batch_size = 100
lr = 0.1
epochs = 10

feature_data = pickle.load(open("mnist_projection.p", "rb"))
x = feature_data["x"]
y = feature_data["y"]

x_placeholder = tf.placeholder(shape=[None, 1024], dtype=tf.float32)
y_placeholder = tf.placeholder(shape=[None, 64], dtype=tf.float32)

y_ = tf.contrib.layers.fully_connected(x_placeholder, 64, activation_fn=None, scope="pca")
error = 0.5*tf.reduce_mean(tf.square(y_ - y_placeholder))
optimizer_op = tf.train.AdadeltaOptimizer(lr).minimize(error)
print(tf.global_variables())

n_batches = len(x) // batch_size

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for j in range(epochs):
	total = 0.0
	for i in range(n_batches):
		loss, _ = sess.run([error, optimizer_op], feed_dict={x_placeholder: x[i*batch_size:(i+1)*batch_size], y_placeholder: y[i*batch_size:(i+1)*batch_size]})
		total += loss
	print(total/n_batches)

with tf.variable_scope("pca", reuse=True):
	diction = {"pca_weights": sess.run(tf.get_variable("weights")), "pca_biases": sess.run(tf.get_variable("biases"))}
	pickle.dump(diction, open("mnist_projection_weights.p", "wb"))
#saver.save(sess, "~/Documents/virtual_emotions/train/mnist_projection/model.ckpt")
#2.576 99% confidence interval