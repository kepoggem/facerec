# pylint: skip-file
""" data iterator for mnist """
import sys

# import the necessary packages
from config import vggface2_config as config
#from neuralnetwork.nn.mxconv import MxVGGNetCl
#from neuralnetwork.nn.mxconv import MxVGGNet
#from vggface2prepare import VGGFace2Prepare
#from neuralnetwork.mxcallbacks import one_off_callback
import mxnet as mx
import argparse
import logging
import pickle
import json
import os
#from neuralnetwork.utils.mxcenter_loss import *

# code to automatically download dataset
mxnet_root = ''
sys.path.append(os.path.join( mxnet_root, 'tests/python/common'))
#import get_data
import mxnet as mx


class custom_mnist_iter(mx.io.DataIter):
	def __init__(self, mnist_iter):
		super(custom_mnist_iter,self).__init__()
		self.data_iter = mnist_iter
		self.batch_size = self.data_iter.batch_size
	
	@property
	def provide_data(self):
		return self.data_iter.provide_data
	
	@property
	def provide_label(self):
		provide_label = self.data_iter.provide_label[0]
		return [('softmax_label', provide_label[1]), \
				('center_label', provide_label[1])]
	
	def hard_reset(self):
		self.data_iter.hard_reset()
	
	def reset(self):
		self.data_iter.reset()
	
	def next(self):
		batch = self.data_iter.next()
		label = batch.label[0]
	
		return mx.io.DataBatch(data=batch.data, label=[label,label], \
				pad=batch.pad, index=batch.index)
	
	

def mnist_iterator(batch_size, input_shape):
	#"""return train and val iterators for mnist"""
	batchSize = 64
	means = json.loads(open(config.DATASET_MEAN).read())
	
	# construct the training image iterator
	train_dataiter = mx.io.ImageRecordIter(
		path_imgrec=config.TRAIN_MX_REC,
		data_shape=(3, 227, 227),
		batch_size=batchSize,
		rand_crop=True,
		rand_mirror=True,
		rotate=7,
		mean_r=means["R"],
		mean_g=means["G"],
		mean_b=means["B"],
		preprocess_threads=config.NUM_DEVICES * 2)
	
	# construct the validation image iterator
	val_dataiter = mx.io.ImageRecordIter(
		path_imgrec=config.VAL_MX_REC,
		data_shape=(3, 227, 227),
		batch_size=batchSize,
		mean_r=means["R"],
		mean_g=means["G"],
		mean_b=means["B"])

	return (custom_mnist_iter(train_dataiter), custom_mnist_iter(val_dataiter))
