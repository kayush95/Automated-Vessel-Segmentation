from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import os
import sys
import numpy as np
#from skimage import io, color, measure
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from predict import *
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score


test_rgb_dir = os.path.join(os.path.curdir,'data','retina','images')
fov_dir = os.path.join(os.path.curdir,'data','retina','mask')
gt_dir = os.path.join(os.path.curdir,'data','retina','1st_manual')

F = tf.app.flags.FLAGS  # for flag handling
batch_size = 64
patch_size = 32
c_dim = 1
output_size = patch_size
experiment_name = 'z_inc_15K_openai'
checkpoint_dir = '/home/15EC90J02/semi_GAN/checkpoint/retina/' + experiment_name
save_threshold = '/home/15EC90J02/semi_GAN/checkpoint/retina/threshold/' + experiment_name
pred_data = '/home/15EC90J02/semi_GAN/checkpoint/retina/threshold/' + experiment_name

def discriminator(image, keep_prob, reuse =None):
  with tf.variable_scope('D'):
	if reuse:
	  tf.get_variable_scope().reuse_variables()

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
	
	h9 = linear(h8, 3, 'd_h9_lin')
	return tf.nn.sigmoid(h9), h9

def fg_prob(probs):
  probability = tf.slice(probs, begin =[0,1], size=[-1,1]) # all rows and second col
  return probability

def load(checkpoint_dir, sess, saver):
  print(" [*] Reading checkpoints...")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
	ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
	print "Succesfully loaded check point files:: "
	return True
  else:
	print "No checkpoint files found"
	return False

def main():
	with tf.Graph().as_default():
		images = tf.placeholder(tf.float32, [batch_size, output_size, output_size, c_dim], name='real_images')
		keep_prob = tf.placeholder(tf.float32)
		D, D_logits = discriminator(images, keep_prob, reuse=None)
		vessel_prob = fg_prob(D)
		print "**Vessel shape: ", vessel_prob.get_shape()
			
		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		with tf.Session() as sess:
			load(checkpoint_dir, sess, saver)  #load saved model 
			test_imgs = [imgs for imgs in os.listdir(test_rgb_dir) if imgs.endswith('clahe_test.tif')] 
			print "Total test images found :  ", len(test_imgs)

			# process each image
			counter = 0

			total_predictions = np.zeros(0)
			total_gt_values = np.zeros(0)

			for im in test_imgs:
				print "Processing test image:  ", im
				img_number = im[0:im.find('_')]
				test = Image.open(os.path.join(test_rgb_dir, im))
				test_img = np.asarray(test, dtype=np.float32)
				test_img = test_img/127.5 - 1  # to make suitable for discriminator in GAN framework   
				gt = Image.open(os.path.join(gt_dir, img_number+'_manual1.gif'))
				gt_image = np.asarray(gt, dtype=np.float32)
				gt_image = gt_image/255.
				fov = Image.open(os.path.join(fov_dir, img_number+'_test_mask.gif'))
				fov_image = np.asarray(fov)
				[ori_H, ori_W] = test_img.shape
				test_img = np.reshape(test_img,(ori_H, ori_W, 1))
				# print "** H and W and C:  ", ori_H, ori_W, test_img.shape[2] 

				test_img_padded = image_pad(test_img, patch_size)
				p_H, p_W, p_C = test_img_padded.shape
				test_patches = extract_test_patches(test_img_padded, patch_size, ori_H, ori_W)
				print "Patch extraction from test image done:", test_patches.shape
				# time.sleep(5)


				gt_img = np.reshape(gt_image,(ori_H, ori_W, 1))
				gt_image_padded = image_pad(gt_img, patch_size)
				assert(p_H == gt_image_padded.shape[0])
				assert(p_W == gt_image_padded.shape[1])
				gt_patches = extract_test_patches(gt_image_padded, patch_size, ori_H, ori_W)
				assert(test_patches.shape == gt_patches.shape)
				print "Patches extracted for gt done: "
				print "Shape of test patches: ", test_patches.shape
				print "Shape of gt_patches: ", gt_patches.shape
				# time.sleep(10)
				print "Range of test and gt", test_img_padded.min(), test_img_padded.max(), gt_image_padded.min(), gt_image_padded.max()
				# Save original and padded images and maps
				t= np.reshape((test_img*127.5 +1.0),(ori_H,ori_W)) 
				g =np.reshape(gt_image, (ori_H, ori_W))
				t_p = np.reshape(test_img_padded*127.5+1, (p_H,p_W))
				g_p = np.reshape(gt_image_padded, (p_H, p_W))

				# print "test image shape*****", t.shape
				# scipy.misc.imsave("test_img.png", t)
				# scipy.misc.imsave("gt_img.png", g)
				# scipy.misc.imsave("test_img_padded.png", t_p)
				# scipy.misc.imsave("gt_img_padded.png", g_p)
				# print "Images dumped success: "

				# ==== make placeholders for storing batchwise prediction results ===============
				predictions = np.zeros((test_patches.shape[0]))
				total_full_batches = int(predictions.shape[0]/ batch_size)
				left_out = predictions.shape[0] % batch_size   # remaining patches in last fractional patch 
				if left_out == 0:
				  pad_req = 0
				else:
				  pad_req = batch_size - left_out  # dummy samples required to make up a batch size
				pad_samples = np.zeros((pad_req,patch_size,patch_size,1))
				print "Total batches, Left Out, Pad_req: ", total_full_batches,left_out, pad_req
				# ==========================================================================================

				# ============= convert gt patches for gt values based on central pixel value
				gt_values = gt_patch_to_value(gt_patches, patch_size, n_samples = predictions.shape[0])
				print gt_values.shape
				assert(gt_values.shape == predictions.shape)
				# ================================================================

				# =============== begin batch wise evaluation ====================
				for batch in range(total_full_batches):
				  image_feed = test_patches[batch * batch_size : batch * batch_size + batch_size, :, :, :]
				  preds = sess.run(vessel_prob, feed_dict={images:image_feed, keep_prob:1.0})

				  if batch%100 ==0:
					print "Batch number processed:", batch
				  predictions[batch*batch_size:batch*batch_size+batch_size] = preds[:,0]

				dummy_feed = np.concatenate((test_patches[-left_out:], pad_samples), axis=0)
				print "Shape of dummy feed is : ", dummy_feed.shape
				mixed_preds = sess.run(vessel_prob, feed_dict={images:dummy_feed, keep_prob:1.0})
				if left_out ==1:
				  predictions[-left_out:] = mixed_preds[0,0]
				else:
				  predictions[-left_out:] = mixed_preds[0:left_out,0]
				print "Predictions Done"

				print "Saving"
				np.save(pred_data + "/" + str(img_number) + "_predictions", predictions, allow_pickle = False)	
				np.save(pred_data + "/" + str(img_number) + "_gt_values", gt_values, allow_pickle = False)				


################################### load ################################################################	
				# predictions = np.load(pred_data + "/" + str(img_number) + "_predictions.npy")
				# gt_values = np.load(pred_data + "/" + str(img_number) + "_gt_values.npy")
#########################################################################################################


				print "Shape of Predictions: ", predictions.shape

				# stich back predictions and gt_values back to image level for comparison ========
				pred_image = recompose(predictions, patch_size, p_H, p_W, ori_H, ori_W)
				gtruth = recompose(gt_values,  patch_size, p_H, p_W, ori_H, ori_W)
				assert(pred_image.shape == gtruth.shape)
				#=========================================================================

				# # ===== saving of Recomposed Images and Maps images==================
				# pi = np.reshape((pred_image*127.5 + 1.0),(p_H,p_W))
				# gtru = np.reshape(gtruth, (p_H,p_W))
				# scipy.misc.imsave(img_number+"pred_image.png", pi)
				# scipy.misc.imsave(img_number+"gtruth_recomposed.png", gtru)

				y_scores, y_true = pred_only_FOV(pred_image, gtruth, fov_image)
				# correct_rate = (y_scores == y_true)
				# correct_rate.astype(np.float32)
				# accuracy_batch = np.mean(correct_rate)
				# print "Classficiation Accurcy****", accuracy_batch

				p = np.random.permutation(len(y_scores))
				y_scores = (y_scores[p])[:30000]
				y_true = (y_true[p])[:30000]

				total_predictions = np.append(total_predictions, y_scores)
				total_gt_values = np.append(total_gt_values, y_true)
				counter += 1

			print "Total Sample: " + str(total_predictions.shape[0])
			assert(total_predictions.shape == total_gt_values.shape)

			maximum = np.ndarray.max(total_predictions)	
			minimum = np.ndarray.min(total_predictions)

			print "Maximum:" + str(maximum)
			print "Minimum:" + str(minimum)

			max_fscore = 0
			best_threshold_fscore = 0

			max_accuracy = 0
			best_threshold_accuracy = 0

			max_specificity = 0
			best_threshold_specificity = 0

			max_sensitivity = 0
			best_threshold_sensitivity = 0

			max_precision = 0
			best_threshold_precision = 0

			max_jaccard = 0 
			best_threshold_jaccard = 0

			c = 0
			for	threshold_confusion in np.linspace(minimum, maximum, 1000):  
				# Confusion matrix
				# threshold_confusion = 0.5
				# print "\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion)
				# y_pred = np.empty((total_predictions.shape[0]))
				c = c + 1
				y_pred = np.copy(total_predictions)
				y_pred[y_pred >= threshold_confusion] = 1
				y_pred[y_pred < threshold_confusion] = 0

				y_true = np.copy(total_gt_values)
				confusion = confusion_matrix(y_true, y_pred)

				# print confusion
				accuracy = 0
				if float(np.sum(confusion))!= 0:
				  accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))

				print "\nGlobal Accuracy: " +str(accuracy)

				specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
				print "Specificity: " +str(specificity)

				sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
				print "Sensitivity: " +str(sensitivity)

				precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
				print "Precision: " +str(precision)

				#Jaccard similarity index
				jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
				print "Jaccard similarity score: " +str(jaccard_index)


				# F1 score
				F1_score = f1_score(total_gt_values, y_pred, labels=None, average='binary', sample_weight=None)
				print "F1 score (F-measure): " +str(F1_score)

				if accuracy > max_accuracy:
					max_accuracy = accuracy
					best_threshold_accuracy = threshold_confusion

				if specificity > max_specificity:
					max_specificity = specificity
					best_threshold_specificity = threshold_confusion

				if sensitivity > max_sensitivity:
					max_sensitivity = sensitivity
					best_threshold_sensitivity = threshold_confusion

				if precision > max_precision:
					max_precision = precision
					best_threshold_precision = threshold_confusion

				if jaccard_index > max_jaccard:
					max_jaccard = jaccard_index
					best_threshold_jaccard = threshold_confusion

				if F1_score > max_fscore:
					max_fscore = F1_score
					best_threshold_fscore = threshold_confusion


			print "Iter: " + str(c)
			# print "Max F1 Score:" + str(max_fscore)
			# Save the results
			print "Saving"

			file_perf = open(save_threshold + '/Threshold_'+ experiment_name +'.txt', 'a')

			file_perf.write("\n Max : "+str(maximum))
			file_perf.write("\n Min : "+str(minimum))

			file_perf.write("\n Max accuracy : "+str(max_accuracy))
			file_perf.write("\n Threshold accuracy : "+str(best_threshold_accuracy))

			file_perf.write("\n Max specificity : "+str(max_specificity))
			file_perf.write("\n Threshold specificity : "+str(best_threshold_specificity))

			file_perf.write("\n Max sensitivity : "+str(max_sensitivity))
			file_perf.write("\n Threshold sensitivity : "+str(best_threshold_sensitivity))

			file_perf.write("\n Max precision : "+str(max_precision))
			file_perf.write("\n Threshold precision : "+str(best_threshold_precision))

			file_perf.write("\n Max jaccard : "+str(max_jaccard))
			file_perf.write("\n Threshold jaccard: "+str(best_threshold_jaccard))

			file_perf.write("\n Max F1 Score : "+str(max_fscore))
			file_perf.write("\n Threshold F1 Score: "+str(best_threshold_fscore))

			file_perf.close()
 
if __name__ == "__main__":
	main()