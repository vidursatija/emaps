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

tf.app.flags.DEFINE_float("learn_rate", 0.15, "Learning rate.")
tf.app.flags.DEFINE_string("pickle_dir", "emotion_data.p", "Data directory")
tf.app.flags.DEFINE_string("model_dir", ".", "Training directory.")
tf.app.flags.DEFINE_string("model_name", "BrainOnMusic", "Model name.")
tf.app.flags.DEFINE_integer("n_steps", 300, "Number of major epochs to run")

FLAGS = tf.app.flags.FLAGS

def train():
	sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)))

	pickle_file = tf.read_file(FLAGS.pickle_dir)
	file_data = pickle.loads(sess.run(pickle_file))
	state_size = 64
	inputX = np.reshape(file_data["x"], [-1, state_size])
	inputCenter = file_data["centers"]
	n_emotions = len(inputCenter) #5
	batch_size = 128
	n_examples = len(inputX)
	n_examples_per_emotion = int(n_examples / n_emotions)
	n_batches = int(n_examples * n_emotions / batch_size)
	#xx = np.reshape(inputX[:n_batches*batch_size], [n_batches, batch_size, state_size])
	#x = np.concatenate([xx for i in range(n_emotions)], 0)
	#cc = np.empty_like(x)
	#yy = np.ones([n_batches*n_emotions, batch_size])
	xx = np.concatenate([inputX for i in range(n_emotions)], 0)
	cc = np.empty_like(xx)
	yy = np.ones([xx.shape[0]])
	for i in range(n_emotions):
		cc[n_examples_per_emotion*i:n_examples_per_emotion*i+n_examples_per_emotion] = np.stack([inputCenter[i] for j in range(n_examples_per_emotion)])
		yy[n_examples_per_emotion*i:n_examples_per_emotion*i+n_examples_per_emotion] -= 1

	x = np.reshape(xx[:n_batches*batch_size], [n_batches, batch_size, state_size])
	c = np.reshape(cc[:n_batches*batch_size], [n_batches, batch_size, state_size])
	y = np.reshape(yy[:n_batches*batch_size], [n_batches, batch_size])

	train_writer = tf.summary.FileWriter(FLAGS.model_dir)

	m = DynamicEmotions(FLAGS.learn_rate)

	sess.run(tf.global_variables_initializer())

	ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Imported")
		tf.logging.info("Imported model")
		m.saver.restore(sess, ckpt.model_checkpoint_path)

	for i in range(FLAGS.n_steps):
		err = m.runEpoch(sess, train_writer, x, c, y, i)
		tf.logging.info(" ".join(["loss:", str(err)]))
		print(" ".join(["loss:", str(err)]))
		if math.isnan(err):
			break
		
		#if i%5 == 4:
		tf.logging.info("Saving model")
		if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
			m.saver.save(sess, ckpt.model_checkpoint_path)
		else:
			m.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
	train_writer.close()


def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
