# USAGE
# python test_preds.py --checkpoints ~/dcnn/datasets/vggface2/checkpoints/ --prefix resnet50 --epoch 20

# import the necessary packages
from config import vggface2_config as config
from neuralnetwork.utils.ranked import rank5_accuracy
from neuralnetwork.utils.mxcenter_loss import *
import mxnet as mx
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
	data = mx.symbol.Variable('data')
	
	softmax_label = mx.symbol.Variable('softmax_label')
	center_label = mx.symbol.Variable('center_label')
	
	# Block #1: (CONV => RELU) * 2 => POOL
	conv1_1 = mx.sym.Convolution(data=data, kernel=(3, 3),
		pad=(1, 1), num_filter=64, name="conv1_1")
	act1_1 = mx.sym.LeakyReLU(data=conv1_1, act_type="prelu",
		name="act1_1")
	bn1_1 = mx.sym.BatchNorm(data=act1_1, name="bn1_1")
	conv1_2 = mx.sym.Convolution(data=bn1_1, kernel=(3, 3),
		pad=(1, 1), num_filter=64, name="conv1_2")
	act1_2 = mx.sym.LeakyReLU(data=conv1_2, act_type="prelu",
		name="act1_2")
	bn1_2 = mx.sym.BatchNorm(data=act1_2, name="bn1_2")
	pool1 = mx.sym.Pooling(data=bn1_2, pool_type="max",
		kernel=(2, 2), stride=(2, 2), name="pool1")
	do1 = mx.sym.Dropout(data=pool1, p=0.25)

	# Block #2: (CONV => RELU) * 2 => POOL
	conv2_1 = mx.sym.Convolution(data=do1, kernel=(3, 3),
		pad=(1, 1), num_filter=128, name="conv2_1")
	act2_1 = mx.sym.LeakyReLU(data=conv2_1, act_type="prelu",
		name="act2_1")
	bn2_1 = mx.sym.BatchNorm(data=act2_1, name="bn2_1")
	conv2_2 = mx.sym.Convolution(data=bn2_1, kernel=(3, 3),
		pad=(1, 1), num_filter=128, name="conv2_2")
	act2_2 = mx.sym.LeakyReLU(data=conv2_2, act_type="prelu",
		name="act2_2")
	bn2_2 = mx.sym.BatchNorm(data=act2_2, name="bn2_2")
	pool2 = mx.sym.Pooling(data=bn2_2, pool_type="max",
		kernel=(2, 2), stride=(2, 2), name="pool2")
	do2 = mx.sym.Dropout(data=pool2, p=0.25)

	# Block #3: (CONV => RELU) * 3 => POOL
	conv3_1 = mx.sym.Convolution(data=do2, kernel=(3, 3),
		pad=(1, 1), num_filter=256, name="conv3_1")
	act3_1 = mx.sym.LeakyReLU(data=conv3_1, act_type="prelu",
		name="act3_1")
	bn3_1 = mx.sym.BatchNorm(data=act3_1, name="bn3_1")
	conv3_2 = mx.sym.Convolution(data=bn3_1, kernel=(3, 3),
		pad=(1, 1), num_filter=256, name="conv3_2")
	act3_2 = mx.sym.LeakyReLU(data=conv3_2, act_type="prelu",
		name="act3_2")
	bn3_2 = mx.sym.BatchNorm(data=act3_2, name="bn3_2")
	conv3_3 = mx.sym.Convolution(data=bn3_2, kernel=(3, 3),
		pad=(1, 1), num_filter=256, name="conv3_3")
	act3_3 = mx.sym.LeakyReLU(data=conv3_3, act_type="prelu",
		name="act3_3")
	bn3_3 = mx.sym.BatchNorm(data=act3_3, name="bn3_3")
	pool3 = mx.sym.Pooling(data=bn3_3, pool_type="max",
		kernel=(2, 2), stride=(2, 2), name="pool3")
	do3 = mx.sym.Dropout(data=pool3, p=0.25)

	# Block #4: (CONV => RELU) * 3 => POOL
	conv4_1 = mx.sym.Convolution(data=do3, kernel=(3, 3),
		pad=(1, 1), num_filter=512, name="conv4_1")
	act4_1 = mx.sym.LeakyReLU(data=conv4_1, act_type="prelu",
		name="act4_1")
	bn4_1 = mx.sym.BatchNorm(data=act4_1, name="bn4_1")
	conv4_2 = mx.sym.Convolution(data=bn4_1, kernel=(3, 3),
		pad=(1, 1), num_filter=512, name="conv4_2")
	act4_2 = mx.sym.LeakyReLU(data=conv4_2, act_type="prelu",
		name="act4_2")
	bn4_2 = mx.sym.BatchNorm(data=act4_2, name="bn4_2")
	conv4_3 = mx.sym.Convolution(data=bn4_2, kernel=(3, 3),
		pad=(1, 1), num_filter=512, name="conv4_3")
	act4_3 = mx.sym.LeakyReLU(data=conv4_3, act_type="prelu",
		name="act4_3")
	bn4_3 = mx.sym.BatchNorm(data=act4_3, name="bn4_3")
	pool4 = mx.sym.Pooling(data=bn4_3, pool_type="max",
		kernel=(2, 2), stride=(2, 2), name="pool3")
	do4 = mx.sym.Dropout(data=pool4, p=0.25)

	# Block #5: (CONV => RELU) * 3 => POOL
	conv5_1 = mx.sym.Convolution(data=do4, kernel=(3, 3),
		pad=(1, 1), num_filter=512, name="conv5_1")
	act5_1 = mx.sym.LeakyReLU(data=conv5_1, act_type="prelu",
		name="act5_1")
	bn5_1 = mx.sym.BatchNorm(data=act5_1, name="bn5_1")
	conv5_2 = mx.sym.Convolution(data=bn5_1, kernel=(3, 3),
		pad=(1, 1), num_filter=512, name="conv5_2")
	act5_2 = mx.sym.LeakyReLU(data=conv5_2, act_type="prelu",
		name="act5_2")
	bn5_2 = mx.sym.BatchNorm(data=act5_2, name="bn5_2")
	conv5_3 = mx.sym.Convolution(data=bn5_2, kernel=(3, 3),
		pad=(1, 1), num_filter=512, name="conv5_3")
	act5_3 = mx.sym.LeakyReLU(data=conv5_3, act_type="prelu",
		name="act5_3")
	bn5_3 = mx.sym.BatchNorm(data=act5_3, name="bn5_3")
	pool5 = mx.sym.Pooling(data=bn5_3, pool_type="max",
		kernel=(2, 2), stride=(2, 2), name="pool5")
	do5 = mx.sym.Dropout(data=pool5, p=0.25)

	# Block #6: FC => RELU layers
	flatten = mx.sym.Flatten(data=do5, name="flatten")
	fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096,
		name="fc1")
	act6_1 = mx.sym.LeakyReLU(data=fc1, act_type="prelu",
		name="act6_1")
	bn6_1 = mx.sym.BatchNorm(data=act6_1, name="bn6_1")
	do6 = mx.sym.Dropout(data=bn6_1, p=0.5)

	# Block #7: FC => RELU layers
	fc2 = mx.sym.FullyConnected(data=do6, num_hidden=4096,
		name="fc2")
	act7_1 = mx.sym.LeakyReLU(data=fc2, act_type="prelu",
		name="act7_1")
	bn7_1 = mx.sym.BatchNorm(data=act7_1, name="bn7_1")
	do7 = mx.sym.Dropout(data=bn7_1, p=0.5)

	# softmax classifier
	
	#embedding = mx.symbol.FullyConnected(data=do7, num_hidden=2, name='embedding')
	
	# second fullc
	fc3 = mx.symbol.FullyConnected(data=do7, num_hidden=config.NUM_CLASSES, name='fc3')
	
	ce_loss = mx.symbol.SoftmaxOutput(data=fc3, label=softmax_label, name='softmax')
	
	center_loss_ = mx.symbol.Custom(data=fc3, label=center_label, name='center_loss_', op_type='centerloss', num_class=config.NUM_CLASSES, alpha=0.5, scale=1.0, batchsize=batchsize)
	center_loss = mx.symbol.MakeLoss(name='center_loss', data=center_loss_)
	mlp = mx.symbol.Group([ce_loss, center_loss])
	#mlp = ce_loss + center_loss
	
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
