
"""  Stop training of D if it becomes too good """
from __future__ import division
import os
import sys
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops_openai import *
from utils_openai import *

import threshold_evaluation_openai 

# from inception_score import inception_score

F = tf.app.flags.FLAGS

class PreGAN(object):
	def __init__(self, sess):
		self.sess = sess
		if F.dataset != "lsun" and F.inc_score:
			print("Loading inception module")
			self.inception_module = inception_score(self.sess)
			print("Inception module loaded")
		self.build_model()

	def build_model(self):
		self.images_labelled = tf.placeholder(tf.float32,
									 [F.batch_size, F.output_size, F.output_size,
									  F.c_dim],
									 name='real_images_l')
		self.images_unlabelled = tf.placeholder(tf.float32,
									 [F.batch_size, F.output_size, F.output_size,
									  F.c_dim],
									 name='real_images_ul')
	   
		self.z_gen = tf.placeholder(tf.float32, [None, F.z_dim], name='z')
		self.labels = tf.placeholder(tf.uint8,
									 [F.batch_size],
									 name='image_labels')
		self.labels_1hot = tf.one_hot(self.labels, depth = F.num_classes)
		  

		self.G_mean, self.G_var, self.G_sample = self.generator(self.z_gen)
		_, self.D_logits_labelled, _ = self.discriminator(self.images_labelled, reuse=False, keep_prob=0.5)
		_, self.D_logits_unlabelled, self.feature_unlabelled = self.discriminator(self.images_unlabelled, reuse=True, keep_prob=0.5)

		_, self.D_logits_fake, self.feature_fake= self.discriminator(self.G_sample, reuse=True, keep_prob=0.5)

		self.z_exp_lab = tf.reduce_mean(log_sum_exp(self.D_logits_labelled))
		self.z_exp_unl = tf.reduce_mean(log_sum_exp(self.D_logits_unlabelled))
		self.z_exp_fake = tf.reduce_mean(log_sum_exp(self.D_logits_fake))

		##### Loss for labelled data

		self.loss_classification = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_labelled, labels = self.labels_1hot))
		self.loss_labelled = self.loss_classification + tf.reduce_mean(self.z_exp_lab)

		#### loss for unlabelled data

		self.l_unl = log_sum_exp(self.D_logits_unlabelled)
		self.loss_unlabelled = -0.5*tf.reduce_mean(self.l_unl) + 0.5*tf.reduce_mean(softplus(log_sum_exp(self.D_logits_unlabelled)))\
													   +  0.5*tf.reduce_mean(softplus(log_sum_exp(self.D_logits_fake)))
	   # Total Discrimninator Los::
		self.d_loss = self.loss_labelled + self.loss_unlabelled

	   # Generator Loss
		self.g_loss_actual = tf.reduce_mean(tf.square(self.feature_unlabelled - self.feature_fake))
			   

		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

	def train(self):
		"""Train DCGAN"""
		data = dataset()
		global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")

		

		learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
													 decay_steps=F.decay_step,
													 decay_rate=F.decay_rate, staircase=True)
		learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
													 decay_steps=F.decay_step,
													 decay_rate=F.decay_rate, staircase=True)
		d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
			.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
			.minimize(self.g_loss_actual, var_list=self.g_vars)

		tf.initialize_all_variables().run()

		

		counter = 0
		start_time = time.time()

		if F.load_chkpt:
			try:
				self.load(F.checkpoint_dir)
				print(" [*] Load SUCCESS")
			except:
				print(" [!] Load failed...")
		else:
			print(" [*] Not Loaded")

		self.ra, self.rb = -1, 1

		for epoch in xrange(1000):
			idx = 0
			iscore = 0.0, 0.0 # self.get_inception_score()
			batch_iter = data.batch()
			for images_l, class_, images_ul  in batch_iter:
				print "Inside!!!!!!!!!!!!!!"
				#sys.exit()
				sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)


				errG_actual = self.g_loss_actual.eval({self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_
										,self.z_gen: sample_z_gen, global_step: epoch})
		
					 
				
				# Update D network
				iters = 1
				if True: 
				  print('Train D net')
				  _,  dlossf = self.sess.run(
					  [d_optim,  self.d_loss],
					  feed_dict={self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_   
								,self.z_gen: sample_z_gen, global_step: epoch})
				  
				 

				# Update G network
				iters = 1
				if True :
				   sample_z_gen = np.random.uniform(self.ra, self.rb,
						[F.batch_size, F.z_dim]).astype(np.float32)
				   print('Train G Net')
				   _,  g_loss_actual = self.sess.run(
						[g_optim,  self.g_loss_actual],
						feed_dict={self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_,
								   self.z_gen: sample_z_gen, global_step: epoch})
				   
					

				loss_l = self.loss_labelled.eval({self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_
								,self.z_gen: sample_z_gen, global_step: epoch})
				loss_ul = self.loss_unlabelled.eval({self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_
								,self.z_gen: sample_z_gen, global_step: epoch})
				g_loss = self.g_loss_actual.eval({self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_
								,self.z_gen: sample_z_gen, global_step: epoch})


				lrateD = learning_rate_D.eval({global_step: epoch})
				lrateG = learning_rate_G.eval({global_step: epoch})
				

				counter += 1
				idx += 1
				print(("Epoch:[%2d] [%4d/%4d] Loss_L:%.2e Loss_UL:%.2e G_Loss:%.8f ")
					  % (epoch, idx, data.num_batches, loss_l, loss_ul, g_loss))



				if np.mod(counter, 100) == 1:
					sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
					samples, g_loss_actual = self.sess.run(
						[self.G_mean,  self.g_loss_actual],
						feed_dict={self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_
								,self.z_gen: sample_z_gen, global_step: epoch}
					)
					save_images(samples, [8, 8],
								F.sample_dir + "/sample.png")
					

				if np.mod(counter, 500) == 2:
					self.save(F.checkpoint_dir)
					print("")
			
			sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
			samples, d_loss, g_loss_actual = self.sess.run(
				[self.G_mean, self.d_loss, self.g_loss_actual],
				feed_dict={ self.images_labelled:images_l, self.images_unlabelled:images_ul, self.labels :class_
								,self.z_gen: sample_z_gen, global_step: epoch}
			)
			save_images(samples, [8, 8],
						F.sample_dir + "/train_{:03d}.png".format(epoch))
			#if epoch % 5 == 0:
			#    iscore = self.get_inception_score()

			if epoch % 10 == 1 :
				threshold_evaluation_openai.main()

	def get_inception_score(self):
		if F.dataset == "lsun" or not F.inc_score:
			return 0.0, 0.0

		samples = []
		for k in range(50000 // F.batch_size):
			sample_z = np.random.uniform(
				self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
			images = self.sess.run(self.G_mean, {self.z: sample_z})
			samples.append(images)
		samples = np.vstack(samples)
		return self.inception_module.get_inception_score(samples)

	def discriminator(self, image, reuse=False, keep_prob=1.0):
		with tf.variable_scope('D'):
			if reuse:
				tf.get_variable_scope().reuse_variables()

			if F.dataset == "celebA":
				dim = 64
				h0 = lrelu(instance_norm(conv2d(image, dim, name='d_h0_conv')))
				h1 = lrelu(instance_norm(conv2d(h0, dim * 2, name='d_h1_conv')))
				h2 = lrelu(instance_norm(conv2d(h1, dim * 4, name='d_h2_conv')))
				h3 = lrelu(instance_norm(conv2d(h2, dim * 8, name='d_h3_conv')))
				h3 = tf.reshape(h3, [F.batch_size, -1])
				#h3 = minibatch_disc(h3, num_kernels=5, scope='d_mb3')
				h4 = linear(h3, 1, 'd_h3_lin')
				return tf.nn.sigmoid(h4), h4

			elif F.dataset == "cifar":
				dim = 96
				noise = tf.random_normal(image.get_shape(), stddev=0.2)
				h0 = lrelu(batch_norm(name='d_bn0')(conv2d(image+noise, dim, 3, 3, 1, 1, name='d_h0_conv')))
				h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim, 3, 3, 1, 1, name='d_h1_conv')))
				h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim, 3, 3, 2, 2, name='d_h2_conv')))
				h2 = tf.nn.dropout(h2, keep_prob=keep_prob)
				h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, 2 * dim, 3, 3, 1, 1, name='d_h3_conv')))
				h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, 2 * dim, 3, 3, 1, 1, name='d_h4_conv')))
				h5 = lrelu(batch_norm(name='d_bn5')(conv2d(h4, 2 * dim, 3, 3, 2, 2, name='d_h5_conv')))
				h5 = tf.nn.dropout(h5, keep_prob=keep_prob)
				h6 = lrelu(batch_norm(name='d_bn6')(conv2d(h5, 2 * dim, 3, 3, 1, 1, name='d_h6_conv')))
				h7 = lrelu(batch_norm(name='d_bn7')(conv2d(h6, 2 * dim, 1, 1, 1, 1, name='d_h7_conv')))
				h8 = tf.reduce_mean(h7, [1, 2])
				#h8 = minibatch_disc(h8, num_kernels=50, scope="d_mb_8")
				h9 = linear(h8, 10, 'd_h9_lin')
				return tf.nn.sigmoid(h9), h9, h8

			elif F.dataset == "retina":
				n_labels = 2
				dim = 96
				h0 = lrelu(instance_norm(conv2d(image, dim, 3, 3, 1, 1, name='d_h0_conv')))
				h1 = lrelu(instance_norm(conv2d(h0, dim, 3, 3, 1, 1, name='d_h1_conv')))
				h2 = lrelu(instance_norm(conv2d(h1, dim, 3, 3, 2, 2, name='d_h2_conv')))
				h2 = tf.nn.dropout(h2, keep_prob)
				h3 = lrelu(instance_norm(conv2d(h2, 2 * dim, 3, 3, 1, 1, name='d_h3_conv')))
				h4 = lrelu(instance_norm(conv2d(h3, 2 * dim, 3, 3, 1, 1, name='d_h4_conv')))
				h5 = lrelu(instance_norm(conv2d(h4, 2 * dim, 3, 3, 2, 2, name='d_h5_conv')))
				h5 = tf.nn.dropout(h5, keep_prob)
				h6 = lrelu(instance_norm(conv2d(h5, 2 * dim, 3, 3, 1, 1, name='d_h6_conv')))
				h7 = lrelu(instance_norm(conv2d(h6, 2 * dim, 1, 1, 1, 1, name='d_h7_conv')))
				h8 = tf.reduce_mean(h7, [1, 2])
				#h8 = minibatch_disc(h8, num_kernels=50, scope="d_mb_8")
				h9 = linear(h8, n_labels+1, 'd_h9_lin')
				return tf.nn.sigmoid(h9), h9, h8

			else:
				#h0 = tf.nn.tanh(linear(tf.reshape(image, [F.batch_size, -1]), 1000, 'd_h0_lin'))
				#h0 = minibatch_disc(h0, scope='d_h0_mbd')
				#h1 = tf.nn.tanh(linear(h0, 1000, 'd_h1_lin'))
				#h2 = linear(h1, 1, 'd_h2_lin')
				#return tf.nn.sigmoid(h2), h2

				print('Shape of Images in D net:::::    ', image.get_shape())
				
				h0_expanded = tf.nn.tanh(linear(tf.reshape(image, [F.batch_size, -1]), 1000, 'd_h0_linn')) #avisek
				

				
				h1_expanded = tf.nn.tanh(linear(h0_expanded,1000,'d_h1_lin_expanded'))

				
				h2_expanded = linear(h1_expanded, 1, 'd_h2_lin_expanded')
				h2_expanded = tf.reshape(h2_expanded,[F.batch_size//F.T,F.T,-1])
				h2 = h2_expanded
				print ('Shape of Noisy samples logit:::', h2_expanded.get_shape())
				return tf.nn.sigmoid(h2), h2



	def generator(self, z):
		with tf.variable_scope("G"):
			if F.dataset == "lsun" or F.dataset == "celebA":
				s = F.output_size
				dim = 64
				s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

				h0 = linear(z, dim * 8 * s16 * s16, scope='g_h0_lin')
				h0 = tf.reshape(h0, [-1, s16, s16, dim * 8])
				h0 = tf.nn.relu(instance_norm(h0))

				h1 = deconv2d(h0, [F.batch_size, s8, s8, dim * 4], name='g_h1')
				h1 = tf.nn.relu(instance_norm(h1))

				h2 = deconv2d(h1, [F.batch_size, s4, s4, dim * 2], name='g_h2')
				h2 = tf.nn.relu(instance_norm(h2))

				h3 = deconv2d(h2, [F.batch_size, s2, s2, dim * 1], name='g_h3')
				h3 = tf.nn.relu(instance_norm(h3))

				h4 = deconv2d(h3, [F.batch_size, s, s, F.c_dim], name='g_h4')
				h4 = tf.nn.tanh(instance_norm(h4))
				mean = h4

				vh0 = linear(z, dim * 4 * s16 * s16, scope='g_vh0_lin')
				vh0 = tf.reshape(vh0, [-1, s16, s16, dim * 4])
				vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

				vh1 = deconv2d(vh0, [F.batch_size, s8, s8, dim * 2], name='g_vh1')
				vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

				vh2 = deconv2d(vh1, [F.batch_size, s4, s4, dim * 1], name='g_vh2')
				vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

				vh3 = deconv2d(vh2, [F.batch_size, s2, s2, dim // 2], name='g_vh3')
				vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))

				vh4 = deconv2d(vh3, [F.batch_size, s, s, F.c_dim], name='g_vh4')
				vh4 = tf.nn.relu(batch_norm(name='g_vbn4')(vh4))
				var = vh4 + F.eps

				eps = tf.random_normal(mean.get_shape())
				sample = mean + eps * var
				return mean, var, sample

			elif F.dataset == "cifar":
				s = F.output_size
				dim = 64
				s2, s4, s8 = int(s / 2), int(s / 4), int(s / 8)

				h0 = linear(z, dim * 8 * s8 * s8, scope='g_h0_lin')
				h0 = tf.reshape(h0, [-1, s8, s8, dim * 8])
				h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0))

				h1 = deconv2d(h0, [F.batch_size, s4, s4, dim * 4], name='g_h1')
				h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

				h2 = deconv2d(h1, [F.batch_size, s2, s2, dim * 2], name='g_h2')
				h2 = tf.nn.relu(batch_norm(name='g_bn2')(h2))

				h3 = deconv2d(h2, [F.batch_size, s, s, F.c_dim], name='g_h3')
				h3 = tf.nn.tanh(batch_norm(name='g_bn3')(h3))
				mean = h3

				vh0 = linear(z, dim * 4 * s8 * s8, scope='g_vh0_lin')
				vh0 = tf.reshape(vh0, [-1, s8, s8, dim * 4])
				vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

				vh1 = deconv2d(vh0, [F.batch_size, s4, s4, dim * 2], name='g_vh1')
				vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

				vh2 = deconv2d(vh1, [F.batch_size, s2, s2, dim], name='g_vh2')
				vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

				vh3 = deconv2d(vh2, [F.batch_size, s, s, F.c_dim], name='g_vh3')
				vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))
				var = vh3 + F.eps

				eps = tf.random_normal(mean.get_shape())
				sample = mean + eps * var
				return mean, var, sample

			elif F.dataset == "retina":
				s = F.output_size
				dim = 64
				s2, s4, s8 = int(s / 2), int(s / 4), int(s / 8)

				h0 = linear(z, dim * 8 * s8 * s8, scope='g_h0_lin')
				h0 = tf.reshape(h0, [-1, s8, s8, dim * 8])
				h0 = tf.nn.relu(instance_norm(h0))

				h1 = deconv2d(h0, [F.batch_size, s4, s4, dim * 4], name='g_h1')
				h1 = tf.nn.relu(instance_norm(h1))

				h2 = deconv2d(h1, [F.batch_size, s2, s2, dim * 2], name='g_h2')
				h2 = tf.nn.relu(instance_norm(h2))

				h3 = deconv2d(h2, [F.batch_size, s, s, F.c_dim], name='g_h3')
				h3 = tf.nn.tanh(instance_norm(h3))
				mean = h3

				vh0 = linear(z, dim * 4 * s8 * s8, scope='g_vh0_lin')
				vh0 = tf.reshape(vh0, [-1, s8, s8, dim * 4])
				vh0 = tf.nn.relu(batch_norm(name='g_vbn0')(vh0))

				vh1 = deconv2d(vh0, [F.batch_size, s4, s4, dim * 2], name='g_vh1')
				vh1 = tf.nn.relu(batch_norm(name='g_vbn1')(vh1))

				vh2 = deconv2d(vh1, [F.batch_size, s2, s2, dim], name='g_vh2')
				vh2 = tf.nn.relu(batch_norm(name='g_vbn2')(vh2))

				vh3 = deconv2d(vh2, [F.batch_size, s, s, F.c_dim], name='g_vh3')
				vh3 = tf.nn.relu(batch_norm(name='g_vbn3')(vh3))
				var = vh3 + F.eps

				eps = tf.random_normal(mean.get_shape())
				sample = mean + eps * var
				return mean, var, sample

			else:
				h0 = tf.nn.tanh(linear(z, 1000, 'g_h0_lin'))
				h1 = tf.nn.tanh(linear(h0, 1000, 'g_h1_lin'))
				h2 = tf.nn.tanh(linear(h1, 784, 'g_h2_lin'))
				mean = h2
				print('MEAN::::::::::::     ', mean.get_shape(), h2.get_shape(), h1.get_shape(),h0.get_shape(), z.get_shape())
				mean = tf.reshape(mean, [F.batch_size//F.T,  28, 28, 1]) #avisek changed

				vh0 = tf.nn.tanh(linear(z, 1000, 'g_vh0_lin'))
				vh1 = tf.nn.tanh(linear(vh0, 1000, 'g_vh1_lin'))
				vh2 = tf.nn.relu(linear(vh1, 784, 'g_vh2_lin', bias_start=F.var))
				var = vh2 + F.eps
				var = tf.reshape(var, [F.batch_size//F.T, 28, 28, 1])   #avisek changed
				eps = tf.random_normal(mean.get_shape())

				sample = mean + eps * var
				print('Important shapes are:: ',eps.get_shape(), mean.get_shape(), var.get_shape(),sample.get_shape())

				mean_expanded = tf.expand_dims(mean,1) # make [B/TX28X28X1] to [B/TX1X28X28X1]
				print('Mean expanded shape::', mean.get_shape())
				mean_expanded = tf.tile(mean_expanded, [1, F.T, 1, 1, 1])
				var_expanded = tf.expand_dims(var,1) # make [B/TX28X28X1] to [B/TX1X28X28X1]
				var_expanded = tf.tile(var_expanded, [1, F.T, 1, 1, 1])
				eps_expanded = tf.random_normal([F.batch_size//F.T, F.T, 28, 28, 1])
				sample_expanded = tf.add(mean_expanded, tf.mul(eps_expanded,var_expanded))
				print ('After changing,::::   ', mean_expanded.get_shape(), var_expanded.get_shape(), sample_expanded.get_shape(),
											   eps_expanded.get_shape()) 
				
				sample_expanded = tf.reshape(sample_expanded,[F.batch_size, 28, 28, 1]) # so that we can feed D net
					

				return mean, var, sample_expanded  #avisek changed sample to sample_extended

	def save(self, checkpoint_dir):
		model_name = "model.ckpt"
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
