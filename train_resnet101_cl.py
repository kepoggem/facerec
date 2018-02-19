import mxnet as mx
import numpy as np
#from center_loss import *
from neuralnetwork.utils.mxcenter_loss import *
from neuralnetwork.nn.mxconv import MxResNet101Cl
from data import mnist_iterator
import logging
import train_model
import argparse
from config import vggface2_config as config

parser = argparse.ArgumentParser(description='train mnist use softmax and centerloss')
parser.add_argument('--gpus', type=str, default='',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--checkpoints', type=str,default='~/dcnn/datasets/vggface2/checkpoints/vgg19cl_checkpoint',
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--start-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log_file', type=str, default='training_vgg19cl_log.log',
                    help='log file')
parser.add_argument('--log_dir', type=str, default='.',
                    help='log dir')
# construct the argument parse and parse the arguments
#parser.add_argument("-c", "--checkpoints", required=True,
#	help="path to output checkpoint directory")
#parser.add_argument("-p", "--prefix", required=True,
#	help="name of model prefix")
args = parser.parse_args()

# mnist input shape
data_shape = (3,227,227)

def get_symbol(batchsize=32):
	"""
	LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
	Haffner. "Gradient-based learning applied to document recognition."
	Proceedings of the IEEE (1998)
	"""
	
	print("[INFO] building network...")
	mlp = MxResNet101Cl.build(config.NUM_CLASSES, (3, 4, 23, 3), (64, 256, 512, 1024, 2048))
		
	return mlp

def main():
	batchsize = args.batch_size if args.gpus is '' else args.batch_size / len(args.gpus.split(','))
	print('batchsize is ', batchsize)
	
	# set the logging level and output file
	if args.start_epoch is None:
		epoch = 0
		logging.basicConfig(level=logging.DEBUG,
		filename="training_resnet50cl_{}.log".format(epoch),
		filemode="w")
	else:
		logging.basicConfig(level=logging.DEBUG,
		filename="training_resnet50cl_{}.log".format(args.start_epoch),
		filemode="w")
	
	# define network structure
	net = get_symbol(batchsize)
	
	# load data
	train, val = mnist_iterator(batch_size=args.batch_size, input_shape=data_shape)
	
	# train
	print('training model ...')
	train_model.fit(args, net, (train, val), data_shape)

if __name__ == "__main__":
	main()
