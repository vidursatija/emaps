from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from sklearn import decomposition as d
from sklearn import cluster as c

from tensorflow.examples.tutorials.mnist import input_data

import mnist_data
import cnn_model

def svd_pca(data, k):
	"""Reduce DATA using its K principal components."""
	data = data.astype("float64")
	data -= np.mean(data, axis=0)
	U, S, V = np.linalg.svd(data, full_matrices=False)
	return U[:,:k].dot(np.diag(S)[:k,:k])

MODEL_DIRECTORY = "model/model02_99.55/"
TEST_BATCH_SIZE = 500

batch_size = TEST_BATCH_SIZE
model_directory = MODEL_DIRECTORY

PIXEL_DEPTH = mnist_data.PIXEL_DEPTH
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#is_training = tf.placeholder(tf.bool, name='MODE')

#x = tf.placeholder(tf.float32, [None, 784])
#y_ = tf.placeholder(tf.float32, [None, 10])  # answer
#fc3, _ = cnn_model.CNN(x, is_training=is_training)

#sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

# Restore variables from disk
#saver = tf.train.Saver()

# Calculate accuracy for all mnist test images
#test_size = mnist.test.num_examples

#saver.restore(sess, model_directory)
checkpoint_file=tf.train.latest_checkpoint(model_directory)
graph=tf.Graph()

with graph.as_default():
	#session_conf = tf.ConfigProto(allow_safe_placement=True, log_device_placement =False)
	sess = tf.Session()#(config = session_conf)
	with sess.as_default():
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess,checkpoint_file)
		#print([n.name for n in graph.as_graph_def().node if n.name[0] == 'f'])#graph.get_collection("model_variables"))
		is_training = graph.get_operation_by_name("MODE").outputs[0]
		x = graph.get_operation_by_name("Placeholder").outputs[0]
		y_ = graph.get_operation_by_name("Placeholder_1").outputs[0]
		fc3 = graph.get_operation_by_name("fc4/Relu").outputs[0]
		n_batches = 50000 // batch_size
		fc3_features = np.empty([n_batches*batch_size, 1024])
		for i in range(n_batches):
			batch = mnist.train.next_batch(batch_size)
			batch_xs = (batch[0] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH  # make zero-centered distribution as in mnist_data.extract_data()
			batch_ys = batch[1]
			fc3_features[i*batch_size:(i+1)*batch_size] = sess.run(fc3, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
			print(i)
		#prediction=graph.get_operation_by_name("prediction").outputs[0]
		#print sess.run(prediction,feed_dict={input:newdata})

emotional_map = svd_pca(fc3_features, 64)

total_std = 1#np.std(emotional_map, axis=0)
total_mean = 0#np.mean(emotional_map, axis=0)

z_map = (emotional_map - total_mean) / total_std
print(z_map.shape)

kmeans = c.KMeans(n_clusters=5)

kmeans.fit(z_map)

diction = {"x": fc3_features, "y": z_map, "centers": kmeans.cluster_centers_}
pickle.dump(diction, open("mnist_projection.p", "wb"))
