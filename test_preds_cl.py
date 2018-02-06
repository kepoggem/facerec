# USAGE
# python test_preds.py --checkpoints ~/dcnn/datasets/vggface2/checkpoints/ --prefix resnet50 --epoch 20

# import the necessary packages
from config import vggface2_config as config
from neuralnetwork.utils.ranked import rank5_accuracy
from neuralnetwork.utils.mxcenter_loss import *
from data import *
import mxnet as mx
import argparse
import pickle
import os
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
	help="epoch # to load")
#args = vars(ap.parse_args())
args = ap.parse_args()


# load model
prefix = args.prefix
model_args = {}
if args.start_epoch is not None:
    assert prefix is not None
    checkpointsPath = os.path.sep.join([args.checkpoints, args.prefix])
    tmp = mx.model.FeedForward.load(checkpointsPath, args.start_epoch)
    #tmp = mx.model.FeedForward.load('/home/kepoggem/dcnn/datasets/vggface2/checkpoints/vgg19cl_pretrained_checkpoints/vgg19', args.start_epoch)
    
    # only add those with the same shape
    arg_dict, aux_dict = get_model_dict( network, data_shape )
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
        if not k in tmp.arg_params.keys():
            continue

        if v == tmp.arg_params[k].shape:
            valid_arg[k] = tmp.arg_params[k]
            print('catching arg: {} from pretrained model'.format(k))
    # for aux 
    for k, v in aux_dict.items():
        # skip these 'label'
        if k == 'data' or k.endswith('label'):
            continue
        
        # skip those pretrain model dosen't have
        if not k in tmp.aux_params.keys():
            continue

        if v == tmp.aux_params[k].shape:
            valid_aux[k] = tmp.aux_params[k]
            print('catching aux: {} from pretrained model'.format(k))

    model_args = {'arg_params' : valid_arg,
                  'aux_params' : valid_aux,
                  'begin_epoch' : args.start_epoch}

# load the label encoder
#le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())

#load R,G,B means
faceMeans = json.loads(open(config.DATASET_MEAN).read())

# construct the validation image iterator
testIter1 = mx.io.ImageRecordIter(
	path_imgrec=config.TEST_MX_REC,
	data_shape=(3, 227, 227),
	batch_size=32,
	mean_r=faceMeans["R"],
	mean_g=faceMeans["G"],
	mean_b=faceMeans["B"])

testIter = custom_mnist_iter(testIter1)

# load our pre-trained model
print("[INFO] loading pre-trained model...")
#checkpointsPath = os.path.sep.join([args["checkpoints"],
#	args["prefix"]])
#(symbol, argParams, auxParams) = mx.model.load_checkpoint(
#	checkpointsPath, args["epoch"])

# construct the model
#model = mx.mod.Module(symbol=symbol, context=[mx.gpu(0)])
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=network,
    #initializer=mx.init.Mixed(init_patterns, init_methods),
    #initializer=mx.initializer.Xavier(),
    #initializer=mx.initializer.MSRAPrelu(),
    #arg_params=argParams,
    #aux_params=auxParams,
    #optimizer=opt,
    #num_epoch=110,
    **model_args)
model.bind(data_shapes=testIter.provide_data, label_shapes=testIter.provide_label)
#model.set_params(argParams, auxParams)
model.set_params(**model_args)

# initialize the list of predictions and targets
print("[INFO] evaluating model...")
predictions = []
targets = []

# loop over the predictions in batches
for (preds, _, batch) in model.iter_predict(testIter):
	# convert the batch of predictions and labels to NumPy
	# arrays
	preds = preds[0].asnumpy()
	labels = batch.label[0].asnumpy().astype("int")

	# update the predictions and targets lists, respectively
	predictions.extend(preds)
	targets.extend(labels)

# apply array slicing to the targets since mxnet will return the
# next full batch size rather than the *actual* number of labels
targets = targets[:len(predictions)]

# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, targets)
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))