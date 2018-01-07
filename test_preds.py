# USAGE
# python test_preds.py --checkpoints ~/dcnn/datasets/vggface2/checkpoints/ --prefix resnet50 --epoch 20

# import the necessary packages
from config import vggface2_config as config
from neuralnetwork.utils.ranked import rank5_accuracy
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

# load our pre-trained model
print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],
	args["prefix"]])
(symbol, argParams, auxParams) = mx.model.load_checkpoint(
	checkpointsPath, args["epoch"])

# construct the model
model = mx.mod.Module(symbol=symbol, context=[mx.gpu(0)])
model.bind(data_shapes=testIter.provide_data,
	label_shapes=testIter.provide_label)
model.set_params(argParams, auxParams)

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