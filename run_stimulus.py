from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import random
import math
from model import DynamicEmotions
from model import StimulusToEmotion

from tensorflow.examples.tutorials.mnist import input_data

import mnist_data

#tf.app.flags.DEFINE_float("learn_rate", 0.15, "Learning rate.")
tf.app.flags.DEFINE_string("pickle_dir", "emotion_data.p", "Data directory")
tf.app.flags.DEFINE_string("model_dir", ".", "Training directory.")
tf.app.flags.DEFINE_string("model_name", "BrainOnMusic", "Model name.")
tf.app.flags.DEFINE_integer("n_steps", 1, "Number of major epochs to run")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))

	state_size = 64

	feature_data = pickle.load(open("mnist_projection.p", "rb"))
	centers = feature_data["centers"]
	n_emotions = len(centers)
	batch_size = 1#n_emotions

	MODEL_DIRECTORY = "model/model02_99.55/"

	model_directory = MODEL_DIRECTORY

	PIXEL_DEPTH = mnist_data.PIXEL_DEPTH
	mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

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
			
			batch = mnist.train.next_batch(batch_size, shuffle=True)
			#centers = mnist.train.next_batch(batch_size, shuffle=True)
			batch_xs = (batch[0] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH  # make zero-centered distribution as in mnist_data.extract_data()
			batch_ys = batch[1]
			#centers_xs = (centers[0] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
			#centers_ys = centers[1]
			fc3_features = sess.run(fc3, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
			#center_features = sess.run(fc3, feed_dict={x: centers_xs, y_: centers_ys, is_training: False})
			STE = StimulusToEmotion("mnist_projection_weights.p", tf.constant(fc3_features), sess)
			DE = DynamicEmotions(0.01, m_pointer=STE.projected_stimulus, location_tvars=STE.get_trainable_vars())
			sess.run(tf.global_variables_initializer())
			initial_m = sess.run(DE.m)
			ee = np.reshape(centers[0], [1, -1])
			for i in range(10):
				reward, distance, emotion_change, ee, _, new_m = sess.run([DE.r, DE.distance, DE.emotion_change, DE.e_plus, DE.location_train_op, DE.m], feed_dict={DE.e: ee})
				delta_m = np.sum(np.square(new_m - initial_m), -1)
				initial_m = new_m
				print(" ".join(["Old Reward", str(np.squeeze(reward)), "New Distance", str(np.squeeze(distance)), "\nChange in E", str(np.squeeze(emotion_change)), "Change in M", str(np.squeeze(delta_m))]))
				print("-----------")
def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
