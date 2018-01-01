from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import os
import random
import math
#from train.model import DynamicEmotions
from model import DynamicEmotions

#tf.app.flags.DEFINE_float("learn_rate", 0.15, "Learning rate.")
tf.app.flags.DEFINE_string("pickle_dir", "emotion_data.p", "Data directory")
tf.app.flags.DEFINE_string("model_dir", ".", "Training directory.")
tf.app.flags.DEFINE_string("model_name", "BrainOnMusic", "Model name.")
tf.app.flags.DEFINE_integer("n_steps", 1, "Number of major epochs to run")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	state_size = 64
	inputX = np.reshape(file_data["x"], [-1, state_size])
	inputCenter = file_data["centers"]
	n_emotions = len(inputCenter) #5
	batch_size = n_emotions
	n_examples = len(inputX)
	n_batches = int(n_examples / batch_size)
	x = np.reshape(inputX[:n_batches*batch_size], [n_batches, batch_size, state_size])
	#centers = np.empty([batch_size, state_size])
	#center_segments = int(batch_size / n_emotions)
	#left_over_segment = batch_size - center_segments*n_emotions
	#for i in range(center_segments):
	#	centers[i*n_emotions:i*n_emotions+n_emotions] = inputCenter
	#centers[-left_over_segment:] = inputCenter[:left_over_segment]

	#train_writer = tf.summary.FileWriter(FLAGS.model_dir)

	m = DynamicEmotions(0.01)

	sess.run(tf.global_variables_initializer())

	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Imported")
		tf.logging.info("Imported model")
		m.saver.restore(sess, ckpt.model_checkpoint_path)

	for i in range(FLAGS.n_steps):
		m.runReal(sess, x[i], inputCenter)
		#err = m.runEpoch(sess, train_writer, x, centers, i)
		#if math.isnan(err):
		#	break
		#tf.logging.info(" ".join(["loss:", str(err)]))
		#print(" ".join(["loss:", str(err)]))
		#if i%5 == 4:
		#tf.logging.info("Saving model")
		#if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		#	m.saver.save(sess, ckpt.model_checkpoint_path)
		#else:
		#	m.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
	#train_writer.close()


def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
