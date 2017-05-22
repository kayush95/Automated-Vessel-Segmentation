import os
import sys
import scipy.misc
import numpy as np
#from model_variable_batch_size import PreGAN
from model_openai import PreGAN
from utils_openai import pp, visualize, to_json

import tensorflow as tf
#from inception_score import inception_score

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100000, "Epochs to train ")
flags.DEFINE_float("learning_rate_D", 0.0001, "Learning rate of adam [0.0001]")
flags.DEFINE_float("learning_rate_G", 0.0001, "Learning rate of adam [0.0001]")
flags.DEFINE_float("beta1D", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("beta1G", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("decay_step",15 , "Decay step of learning rate in epochs")
flags.DEFINE_float("decay_rate", 0.7, "Decay rate of learning rate")
flags.DEFINE_float("eps", 1e-7, "Epsilon")
flags.DEFINE_float("var", 0.01, "Variance")
flags.DEFINE_float("gpu_frac", 0.5, "Gpu fraction")

dataset = "retina"
comment = "z_inc_15K_openai"
flags.DEFINE_string("dataset_size", 15000, "No. of labelled examples of each vessel and background patches")

flags.DEFINE_string("dataset", dataset, "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("data_dir", "data/" + dataset, "Directory name containing the dataset [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint/" + dataset+ "/"+ comment, "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples/" + dataset + "/"+ comment, "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs/" + dataset + "/"+ comment, "Directory name to save the logs [logs]")
flags.DEFINE_boolean("load_chkpt", False, "True for loading saved checkpoint")
flags.DEFINE_boolean("inc_score", False, "True for computing inception score")

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("z_dim", 300, "Dimension of latent vector.")


if dataset == "mnist":
    flags.DEFINE_integer("c_dim", 1, "Number of channels in input image")
    flags.DEFINE_boolean("is_grayscale", True, "True for grayscale image")
    flags.DEFINE_integer("output_size", 28, "True for grayscale image")
    flags.DEFINE_integer("num_classes", 10, "Number of output classes")

elif dataset == "lsun" or dataset == "lfw":
    flags.DEFINE_integer("c_dim", 3, "Number of channels in input image")
    flags.DEFINE_boolean("is_grayscale", False, "True for grayscale image")
    flags.DEFINE_integer("output_size", 64, "True for grayscale image")
    flags.DEFINE_integer("num_classes", 3, "Number of output classes")

elif dataset == "retina":
    flags.DEFINE_integer("c_dim", 1, "Number of channels in input image")
    flags.DEFINE_boolean("is_grayscale", True, "True for grayscale image")
    flags.DEFINE_integer("output_size", 32, "True for grayscale image")	
    flags.DEFINE_integer("num_classes", 3, "Number of output classes")

else:
    flags.DEFINE_integer("c_dim", 3, "Number of channels in input image")
    flags.DEFINE_boolean("is_grayscale", False, "True for grayscale image")
    flags.DEFINE_integer("output_size", 32, "True for grayscale image")
    flags.DEFINE_integer("num_classes", 3, "Number of output classes")

FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # with tf.Session() as sess:
        dcgan = PreGAN(sess)
        dcgan.train()

if __name__ == '__main__':
    tf.app.run()

