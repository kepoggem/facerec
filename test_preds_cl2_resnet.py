# USAGE
# python test_preds.py --checkpoints ~/dcnn/datasets/vggface2/checkpoints/ --prefix resnet50 --epoch 20

# import the necessary packages
from config import vggface2_config as config
from neuralnetwork.utils.ranked import rank5_accuracy
from neuralnetwork.utils.mxcenter_loss import *
import mxnet as mx
from neuralnetwork.nn.mxconv import MxResNetCl
from config import vggface2_config as config
from data import mnist_iterator
#from train import get_symbol
from train_model import get_model_dict
import argparse
import pickle
import os
import json

def get_symbol(batchsize=32):
	"""
	LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
	Haffner. "Gradient-based learning applied to document recognition."
	Proceedings of the IEEE (1998)
	"""
	print("[INFO] building network...")
	mlp = MxResNetCl.build(config.NUM_CLASSES, (3, 4, 6, 3), (64, 256, 512, 1024, 2048))
		
	return mlp


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
	help="epoch # to load")
args = vars(ap.parse_args())

# load the label encoder
#le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())

#load R,G,B means
faceMeans = json.loads(open(config.DATASET_MEAN).read())

# construct the validation image iterator
testIter = mx.io.ImageRecordIter(
	path_imgrec=config.TEST_MX_REC,
	data_shape=(3, 227, 227),
	batch_size=32,
	mean_r=faceMeans["R"],
	mean_g=faceMeans["G"],
	mean_b=faceMeans["B"])

## custom
network = get_symbol(32)

# load our pre-trained model
print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
#(symbol, argParams, auxParams) = mx.model.load_checkpoint(checkpointsPath, args["epoch"])
symbol = mx.model.FeedForward.load(checkpointsPath, args["epoch"])

arg_dict, aux_dict = get_model_dict( network, (3,227,227) )
valid_arg = dict()
valid_aux = dict()

# print all the parameters
print('all params ', arg_dict)

# for args 
for k, v in arg_dict.items():
    # skip those 'label'
    if k == 'data' or k.endswith('label'):
        continue

    # skip those pretrain model dosen't have
    if not k in symbol.arg_params.keys():
        continue

    if v == symbol.arg_params[k].shape:
        valid_arg[k] = symbol.arg_params[k]
        print('catching arg: {} from pretrained model'.format(k))
# for aux 
for k, v in aux_dict.items():
    # skip these 'label'
    if k == 'data' or k.endswith('label'):
        continue
    
    # skip those pretrain model dosen't have
    if not k in symbol.aux_params.keys():
        continue

    if v == symbol.aux_params[k].shape:
        valid_aux[k] = symbol.aux_params[k]
        print('catching aux: {} from pretrained model'.format(k))

model_args = {'arg_params' : valid_arg,
              'aux_params' : valid_aux,
              'begin_epoch' : args["epoch"]}

train, val = mnist_iterator(batch_size=32, input_shape=(3,227,227))
# construct the model
#model = mx.model.FeedForward(ctx=[mx.gpu(0), mx.gpu(1)], symbol=network, **model_args)
model = mx.mod.Module(symbol=network, context=[mx.gpu(0), mx.gpu(1)])
model.bind(data_shapes=val.provide_data, label_shapes=val.provide_label)
#model.set_params(argParams, auxParams, allow_missing=True)
model.set_params(valid_arg, valid_aux, allow_missing=True)

# initialize the list of predictions and targets
print("[INFO] evaluating model...")
predictions = []
targets = []

# loop over the predictions in batches
for (preds, i_batch, batch) in model.iter_predict(testIter):
#	# convert the batch of predictions and labels to NumPy
#	# arrays
	preds = preds[0].asnumpy()
	labels = batch.label[0].asnumpy().astype("int")
	if i_batch == 503:
		print("[INFO] i_batch number is {}.".format(i_batch))
		break
	# update the predictions and targets lists, respectively
	predictions.extend(preds)
	targets.extend(labels)

# apply array slicing to the targets since mxnet will return the
# next full batch size rather than the *actual* number of labels
print("[INFO] Results:")
targets = targets[:len(predictions)]

# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, targets)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
