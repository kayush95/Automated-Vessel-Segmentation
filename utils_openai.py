"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import os, sys
import json
import random
import pprint
import scipy.misc
import numpy as np
# import lmdb
# import cv2
import tensorflow as tf
from time import gmtime, strftime
import glob

pp = pprint.PrettyPrinter()
F = tf.app.flags.FLAGS

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_image(image_path, image_size, is_crop=True,resize_w=32, is_grayscale=False):
	return transform(imread(image_path, is_grayscale), image_size, is_crop)


def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
	if (is_grayscale):
		return scipy.misc.imread(path, flatten=True).astype(np.float)
	else:
		return scipy.misc.imread(path).astype(np.float)


class dataset(object):
	def __init__(self):
		if F.dataset == 'mnist':
			self.data = load_mnist()
		elif F.dataset == 'lsun':
			self.data = lmdb.open(F.data_dir, map_size=500000000000,  # 500 gigabytes
								  max_readers=100, readonly=True)
		elif F.dataset == 'cifar':
			self.data, self.labels, self.data_unlabelled = load_cifar()
		elif F.dataset == 'celebA':
			self.data = load_celebA()
		
		elif F.dataset == 'retina':
			self.data_lab, self.label, self.data_unlab = load_retina()

		else:
			raise NotImplementedError("Does not support dataset {}".format(F.dataset))

	def batch(self):
		if F.dataset == 'retina':
			self.num_batches = len(self.data_lab) // F.batch_size
			for i in range(self.num_batches):
				yield self.data_lab[i * F.batch_size:(i + 1) * F.batch_size], self.label[i * F.batch_size:(i + 1) * F.batch_size], self.data_unlab[i * F.batch_size:(i + 1) * F.batch_size]


		elif F.dataset != 'lsun' and F.dataset != 'celebA':
			print "Inside batch size!!!"
			self.num_batches = len(self.data) // F.batch_size
			print "Total number of train", len(self.data)
			print "Total number of test:", len(self.data_unlabelled)
			for i in range(self.num_batches):
				start, end = i * F.batch_size, (i + 1) * F.batch_size
				#if end > len(self.data_unlabelled):
				#  start_ul = 0
				  #end_ul = 
				yield self.data[start:end], self.labels[start:end] , self.data_unlabelled[start:end]

				

		elif F.dataset == 'lsun':
			self.num_batches = 3033042 // F.batch_size
			with self.data.begin(write=False) as txn:
				cursor = txn.cursor()
				examples = 0
				batch = []
				for key, val in cursor:
					img = np.fromstring(val, dtype=np.uint8)
					img = cv2.imdecode(img, 1)
					img = transform(img)
					batch.append(img)
					examples += 1

					if examples >= F.batch_size:
						batch = np.asarray(batch)[:, :, :, [2, 1, 0]]
						yield batch
						batch = []
						examples = 0
	   
		else:
		  self.num_batches = len(self.data)// F.batch_size
		  print "**************Number of batches will be ::", self.num_batches
		  
		  for i in range(self.num_batches): 
			sub_list = self.data[i * F.batch_size:(i + 1) * F.batch_size] 
			image_list = []
			for item in sub_list: 
			  image_list.append(get_image(image_path=os.path.join(F.data_dir, item), 
											image_size=F.output_size)) 
			sample_images = np.asarray(image_list)
			print "*** Before yielding shape: ", sample_images.shape
			yield sample_images


def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def load_retina():
	bg1 = glob.glob(os.path.join("../../data", F.dataset, "Adversarial_BG1", "*.png"))
	bg2 = glob.glob(os.path.join("../../data", F.dataset, "Adversarial_BG2", "*.png"))
	bg3 = glob.glob(os.path.join("../../data", F.dataset, "Adversarial_BG3", "*.png"))

	fg1 = glob.glob(os.path.join("../../data", F.dataset, "Adversarial_FG1", "*.png"))
	fg2 = glob.glob(os.path.join("../../data", F.dataset, "Adversarial_FG2", "*.png"))
	fg3 = glob.glob(os.path.join("../../data", F.dataset, "Adversarial_FG3", "*.png"))


	bg_full = bg1 + bg2 + bg3
	fg_full = fg1 + fg2 + fg3

	dataset_size = F.dataset_size

	bg_lab = bg_full[:dataset_size]
	bg_unlab = bg_full[dataset_size:]

	fg_lab = fg_full[:dataset_size]
	fg_unlab = fg_full[dataset_size:]

	rem = len(bg_lab)%dataset_size
	temp = bg_lab[:rem]
	bg_lab = bg_lab*(int(len(bg_full)/dataset_size) - 1) + temp

	rem = len(fg_lab)%dataset_size
	temp = fg_lab[:rem]
	fg_lab = fg_lab*(int(len(fg_full)/dataset_size) - 1) + temp

	print "Glob reading done:  "
	# bg = [0]*(len(bg1)+ len(bg2) + len(bg3))
	# fg = [1]*(len(fg1)+ len(fg2) + len(fg3))
	bg = [0]*(len(bg_lab))
	fg = [1]*(len(fg_lab))

	labels = bg + fg
	labels = np.asarray(labels, dtype=np.uint8)
	print "Label creation done"

	data_raw_lab = bg_lab + fg_lab
	data_files = data_raw_lab#[0:1025]
	data = [get_image(data_file, 32, is_crop=False, resize_w=32, is_grayscale = True) for data_file in data_files]
	X_lab = np.array(data).astype(np.float32)[:, :, :, None]
	print "Data array created (labelled)"

	data_raw_unlab = bg_unlab + fg_unlab
	data_files = data_raw_unlab#[0:1025]
	data = [get_image(data_file, 32, is_crop=False, resize_w=32, is_grayscale = True) for data_file in data_files]
	X_unlab = np.array(data).astype(np.float32)[:, :, :, None]
	print "Data array created (unlabelled)"    
	
	X_lab, Y = unison_shuffled_copies(X_lab, labels)
	print "Labels are:: ", Y[1:30]

	p = np.random.permutation(len(X_unlab))
	X_unlab = X_unlab[p]

	return X_lab, Y, X_unlab


def load_mnist():   
	fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	trY = loaded[8:].reshape((60000)).astype(np.float)

	fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

	fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	teY = loaded[8:].reshape((10000)).astype(np.float)

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0)

	X = X / 127.5 - 1

	combined = zip(X, labels)
	np.random.shuffle(combined)

	X = zip(*combined)[0]
	y = zip(*combined)[1]

	return X, y


def load_cifar():
	def unpickle(file):
		import cPickle
		fo = open(file, 'rb')
		dict = cPickle.load(fo)
		fo.close()
		return dict

	trainX1 = unpickle(F.data_dir + '/data_batch_1')
	trainX2 = unpickle(F.data_dir + '/data_batch_2')
	trainX3 = unpickle(F.data_dir + '/data_batch_3')
	trainX4 = unpickle(F.data_dir + '/data_batch_4')
	trainX5 = unpickle(F.data_dir + '/data_batch_5')

	trainX = np.vstack((trainX1['data'], trainX2['data'], trainX3[
					   'data'], trainX4['data'], trainX5['data']))

	trainX = np.asarray(trainX / 127.5, dtype=np.float32) - 1.0
	trainX = trainX.reshape(-1, 3, 32, 32)
	trainX = trainX.transpose(0, 2, 3, 1)
		 

	trainY = np.hstack((trainX1['labels'], trainX2['labels'], trainX3[
					   'labels'], trainX4['labels'], trainX5['labels']))
	print "Shape of labels:", trainY.shape
	
	label_map = {}

	def map_label(label):
		if label not in label_map:
			label_map[label] = len(label_map)
		return label_map[label]

	for i in range(len(trainY)):
		trainY[i] = map_label(trainY[i])
	trainY = np.asarray(trainY, dtype=np.uint8)

	perm = np.random.permutation(trainX.shape[0])
	trainX = trainX[perm]
	trainY = trainY[perm]
	
	data_labelled = trainX[0:10000]
	labels = trainY[0:10000]
	data_unlabelled = trainX[10001:30001]
	print "*****Shapes are", data_labelled.shape, labels.shape, data_unlabelled.shape
	sys.exit()
	return trainX[0:25000], trainY[0:25000], trainX[25001:49999]


	

def load_celebA():
  file_list = os.listdir(F.data_dir)
  print "Total images found: ", len(file_list)
  file_list = np.random.permutation(file_list)
  return file_list

def merge_images(images, size):
	return inverse_transform(images)


def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j * h: j * h + h, i * w: i * w + w, :] = image

	return img


def imsave(images, size, path):
	return scipy.misc.toimage(merge(images, size), cmax=1.0, cmin=0.0).save(path)


def square_crop(x, npx):
	h, w = x.shape[:2]
	crop_size = min(h, w)
	i = int((h - crop_size) / 2.)
	j = int((w - crop_size) / 2.)
	return scipy.misc.imresize(x[i:i + crop_size, j:j + crop_size],
							   [npx, npx])


def transform(image, npx=64, is_crop=True):
	# npx : # of pixels width/height of image
	if is_crop:
		cropped_image = square_crop(image, npx)
	else:
		cropped_image = image
	return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
	return (images + 1.) / 2.


def vgg16(input, name="vgg16"):
	with tf.variable_scope(name) as scope:
		scope_name = tf.get_variable_scope().name

		shp = input.get_shape().as_list()
		if len(shp) == 3:
			input = tf.expand_dims(input, -1)
		images = tf.image.resize_bicubic(input, [224, 224], align_corners=None, name=None)

		with open("vgg16.tfmodel", mode='rb') as f:
			fileContent = f.read()

		graph_def = tf.GraphDef()
		graph_def.ParseFromString(fileContent)

		tf.import_graph_def(graph_def, input_map={"images": images})
		print("Graph loaded from disk")

		graph = tf.get_default_graph()
		# for op in self.graph.get_operations():
		#   print(op.values())

		feats = graph.get_tensor_by_name(scope_name + "/import/pool4:0")
		return feats


def to_json(output_path, *layers):
	with open(output_path, "w") as layer_f:
		lines = ""
		for w, b, bn in layers:
			layer_idx = w.name.split('/')[0].split('h')[1]

			B = b.eval()

			if "lin/" in w.name:
				W = w.eval()
				depth = W.shape[1]
			else:
				W = np.rollaxis(w.eval(), 2, 0)
				depth = W.shape[0]

			biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
			if bn is not None:
				gamma = bn.gamma.eval()
				beta = bn.beta.eval()

				gamma = {"sy": 1, "sx": 1, "depth": depth, "w": [
					'%.2f' % elem for elem in list(gamma)]}
				beta = {"sy": 1, "sx": 1, "depth": depth, "w": [
					'%.2f' % elem for elem in list(beta)]}
			else:
				gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
				beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

			if "lin/" in w.name:
				fs = []
				for w in W.T:
					fs.append({"sy": 1, "sx": 1, "depth": W.shape[
							  0], "w": ['%.2f' % elem for elem in list(w)]})

				lines += """
					var layer_%s = {
						"layer_type": "fc",
						"sy": 1, "sx": 1,
						"out_sx": 1, "out_sy": 1,
						"stride": 1, "pad": 0,
						"out_depth": %s, "in_depth": %s,
						"biases": %s,
						"gamma": %s,
						"beta": %s,
						"filters": %s
					};""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
			else:
				fs = []
				for w_ in W:
					fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": [
							  '%.2f' % elem for elem in list(w_.flatten())]})

				lines += """
					var layer_%s = {
						"layer_type": "deconv",
						"sy": 5, "sx": 5,
						"out_sx": %s, "out_sy": %s,
						"stride": 2, "pad": 1,
						"out_depth": %s, "in_depth": %s,
						"biases": %s,
						"gamma": %s,
						"beta": %s,
						"filters": %s
					};""" % (layer_idx, 2**(int(layer_idx) + 2), 2**(int(layer_idx) + 2),
							 W.shape[0], W.shape[3], biases, gamma, beta, fs)
		layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
	import moviepy.editor as mpy

	def make_frame(t):
		try:
			x = images[int(len(images) / duration * t)]
		except:
			x = images[-1]

		if true_image:
			return x.astype(np.uint8)
		else:
			return ((x + 1) / 2 * 255).astype(np.uint8)

	clip = mpy.VideoClip(make_frame, duration=duration)
	clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
	if option == 0:
		z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
		samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
		save_images(samples, [8, 8], './samples/test_%s.png' %
					strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	elif option == 1:
		values = np.arange(0, 1, 1. / config.batch_size)
		for idx in xrange(100):
			print(" [*] %d" % idx)
			z_sample = np.zeros([config.batch_size, dcgan.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
			save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
	elif option == 2:
		values = np.arange(0, 1, 1. / config.batch_size)
		for idx in [random.randint(0, 99) for _ in xrange(100)]:
			print(" [*] %d" % idx)
			z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
			z_sample = np.tile(z, (config.batch_size, 1))
			# z_sample = np.zeros([config.batch_size, dcgan.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
			make_gif(samples, './samples/test_gif_%s.gif' % (idx))
	elif option == 3:
		values = np.arange(0, 1, 1. / config.batch_size)
		for idx in xrange(100):
			print(" [*] %d" % idx)
			z_sample = np.zeros([config.batch_size, dcgan.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
			make_gif(samples, './samples/test_gif_%s.gif' % (idx))
	elif option == 4:
		image_set = []
		values = np.arange(0, 1, 1. / config.batch_size)

		for idx in xrange(100):
			print(" [*] %d" % idx)
			z_sample = np.zeros([config.batch_size, dcgan.z_dim])
			for kdx, z in enumerate(z_sample):
				z[idx] = values[kdx]

			image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
			make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

		new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10])
						 for idx in range(64) + range(63, -1, -1)]
		make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
