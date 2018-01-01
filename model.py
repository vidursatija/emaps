from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle

class DynamicEmotions():
	def __init__(self, learn_rate, m_pointer=None, location_tvars=None, state_size=64, margin=4.0):
		self.state_size = state_size
		self.learn_rate = tf.Variable(tf.constant(learn_rate), name="rate_learn")
		self.importance = 0
		self.margin = margin

		self.m = m_pointer# or tf.placeholder(shape=[None, state_size], dtype=tf.float32)
		self.e = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
		#self.y = tf.placeholder(shape=[None], dtype=tf.float32)
		#self.alpha = tf.placeholder(shape=(), dtype=tf.float32)

		self.r, self.e_plus = self.emotionBlock(self.m, self.e)

		#self.r_plus, self.e_plus_plus = self.emotionBlock(self.m, self.e_plus, reuse=True)

		optimizer = tf.train.AdadeltaOptimizer(self.learn_rate)
		self.distance = tf.reduce_sum(tf.square(self.m-self.e_plus), -1)
		self.emotion_change = tf.reduce_sum(tf.square(self.e-self.e_plus), -1)
		#self.distance_plus = tf.reduce_mean(tf.square(self.e_plus_plus-self.m), -1)
		#if location_tvars is not None:
		self.location_loss = 0.5*tf.reduce_mean(self.distance)#0.5*tf.reduce_mean((1-self.y)*self.importance*(tf.reduce_mean(tf.square(self.e-self.e_plus), -1)) + self.distance)/(1+self.importance) + 0.5*tf.reduce_mean((1-self.y)*self.importance*(tf.reduce_mean(tf.square(self.e_plus-self.e_plus_plus), -1)) + self.distance_plus)/(1+self.importance)
		#location_tvars = [v for v in tf.global_variables() if v.name[0] == "l"]
		location_gradsvars = optimizer.compute_gradients(self.location_loss, var_list=location_tvars)
		location_grads, _ = tf.clip_by_global_norm([g for g, v in location_gradsvars], 10)
		self.location_train_op = optimizer.apply_gradients(zip(location_grads, location_tvars))

		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_assign = tf.assign(self.learn_rate, self.new_lr)

		#self.saver = tf.train.Saver([v for v in tf.global_variables() if v.name != "rate_learn"])
		#self.summary = tf.summary.scalar('location-loss', self.location_loss)

	def emotionBlock(self, m, e, reuse=False, alpha=0.01):
		merged_me = tf.concat([m, e], -1)
		d = tf.sqrt(tf.reduce_sum(tf.square(m-e), -1, keep_dims=True))
		r = tf.minimum(1., 2.25/tf.square(d/self.margin+1)) #tf.contrib.layers.fully_connected(merged_me, self.state_size, scope="l_r", activation_fn=tf.sigmoid, reuse=reuse)
		#i = tf.contrib.layers.fully_connected(merged_me, self.state_size, scope="l_i", activation_fn=tf.sigmoid, reuse=reuse)
		variability = m-e#tf.contrib.layers.fully_connected(merged_me, self.state_size, scope="l_v", activation_fn=tf.tanh, reuse=reuse)
		e_plus = e + (1-r)*variability*alpha
		return r, e_plus

	def runEpoch(self, sess, train_writer, x, centers, y, epoch_num):
		#print(x.shape)
		#print(centers.shape)
		#print(y.shape)
		n_examples = len(x)
		total_error = 0.0
		for i in range(n_examples):
			summary, loss, _ = sess.run([self.summary, self.location_loss, self.location_train_op], feed_dict={self.e: centers[i]})
			train_writer.add_summary(summary, epoch_num*n_examples + i)
			total_error += loss
		m_batch = centers
		#c_lr = sess.run(self.learn_rate)
		#sess.run(self.lr_assign, feed_dict={self.new_lr:c_lr*0.99})
		return total_error / (n_examples)

	def runReal(self, sess, x, centers, steps=10):
		#print(x.shape)
		#print(centers.shape)
		n_examples = steps
		n_emotions = len(centers)
		m_batch = centers
		ex = 1*np.random.rand(1, self.state_size) - .5
		example = np.concatenate([ex for i in range(n_emotions)])
		print(np.sum(np.square(example-m_batch), -1))
		for i in range(n_examples):
			distance, r_array, m_batch = sess.run([self.distance, self.r, self.e_plus], feed_dict={self.m: example, self.e: m_batch, self.alpha: 1.0})
			for j in range(n_emotions):
				print(" ".join(["Emotion", str(j), "Distance:", str(distance[j]), "Reward:", str(np.mean(r_array[j]))]))
			print("-----")

class StimulusToEmotion():
	def __init__(self, projection_pickle_dir, stimulus, sess, state_size=64):
		variable_data = pickle.load(open(projection_pickle_dir, "rb"))
		w = variable_data["pca_weights"]
		b = variable_data["pca_biases"]
		self.projected_stimulus = tf.contrib.layers.fully_connected(stimulus, state_size, activation_fn=None, scope="pca")
		with tf.variable_scope("pca", reuse=True) as scope:
			self.w = tf.get_variable("weights")
			self.b = tf.get_variable("biases")
		uw = tf.assign(self.w, w)
		ub = tf.assign(self.b, b)
		sess.run([uw, ub])

	def get_trainable_vars(self):
		return (self.w, self.b)

