# USAGE
# python plot_log.py --network VGGNet --dataset ImageNet

# import the necessary packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True,
	help="name of network")
ap.add_argument("-d", "--dataset", required=True,
	help="name of dataset")
args = vars(ap.parse_args())

# define the paths to the training logs
logs = [
	(15, "training_resnet101_0.log"),
	(17, "training_resnet101_15.log"),
	(20, "training_resnet101_17.log"),
	(24, "training_resnet101_20.log")
]

# initialize the list of train rank-1 and rank-5 accuracies, along
# with the training loss
(trainRank1, trainLoss) = ([], [])

# initialize the list of validation rank-1 and rank-5 accuracies,
# along with the validation loss
(valRank1, valLoss) = ([], [])

# loop over the training logs
for (i, (endEpoch, p)) in enumerate(logs):
	# load the contents of the log file, then initialize the batch
	# lists for the training and validation data
	rows = open(p).read().strip()
	(bTrainRank1, bTrainLoss) = ([], [])
	(bValRank1, bValLoss) = ([], [])

	# grab the set of training epochs
	epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
	epochs = sorted([int(e) for e in epochs])

	# loop over the epochs
	for e in epochs:
		# find all rank-1 accuracies, rank-5 accuracies, and loss
		# values, then take the final entry in the list for each
		s = r'Epoch\[' + str(e) + '\].*accuracy=([0]*\.?[0-9]+)'
		rank1 = re.findall(s, rows)[-2]
		#s = r'Epoch\[' + str(e) + '\].*top_k_accuracy_5=([0]*\.?[0-9]+)'
		#rank5 = re.findall(s, rows)[-2]
		s = r'Epoch\[' + str(e) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
		loss = re.findall(s, rows)[-2]

		# update the batch training lists
		bTrainRank1.append(float(rank1))
		#bTrainRank5.append(float(rank5))
		bTrainLoss.append(float(loss))
	
	# extract the validation rank-1 and rank-5 accuracies for each
	# epoch, followed by the loss
	bValRank1 = re.findall(r'Validation-accuracy=(.*)', rows)
	#bValRank5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
	bValLoss = re.findall(r'Validation-cross-entropy=(.*)', rows)

	# convert the validation rank-1, rank-5, and loss lists to floats
	bValRank1 = [float(x) for x in bValRank1]
	#bValRank5 = [float(x) for x in bValRank5]
	bValLoss = [float(x) for x in bValLoss]

	# check to see if we are examining a log file other than the
	# first one, and if so, use the number of the final epoch in
	# the log file as our slice index
	if i > 0 and endEpoch is not None:
		trainEnd = endEpoch - logs[i - 1][0]
		valEnd = endEpoch - logs[i - 1][0]

	# otherwise, this is the first epoch so no subtraction needs
	# to be done
	else:
		trainEnd = endEpoch
		valEnd = endEpoch

	# update the training lists
	trainRank1.extend(bTrainRank1[0:trainEnd])
	#trainRank5.extend(bTrainRank5[0:trainEnd])
	trainLoss.extend(bTrainLoss[0:trainEnd])

	# update the validation lists
	valRank1.extend(bValRank1[0:valEnd])
	#valRank5.extend(bValRank5[0:valEnd])
	valLoss.extend(bValLoss[0:valEnd])

# plot the accuracies
plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0, len(trainRank5)), trainRank5,
#	label="train_rank5")
plt.plot(np.arange(0, len(valRank1)), valRank1, label="val_acc")
#plt.plot(np.arange(0, len(valRank5)), valRank5,
#	label="val_rank5")
plt.plot(np.arange(0, len(trainRank1)), trainRank1, label="train_acc")
plt.title("{}: accuracy on {}".format(args["network"], args["dataset"]))
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig('acc.png')

# plot the losses
plt.style.use("ggplot")
#plt.figure()
#fig, ax1 = plt.subplots()
fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = ax1.twinx()
tl = ax1.plot(np.arange(0, len(trainLoss)), trainLoss, label="train_loss")
vl = ax1.plot(np.arange(0, len(valLoss)), valLoss, label="val_loss")
ax2 = ax1.twinx()   # mirror them
ta = ax2.plot(np.arange(0, len(trainRank1)), trainRank1, 'tab:purple', label="train_acc")
va = ax2.plot(np.arange(0, len(valRank1)), valRank1, 'tab:gray', label="val_acc")

tsa = ax2.axhline(y=0.9418, color='g', linestyle=':', label="test_acc")

lns = tl+vl+ta+va
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc="upper left")
plt.grid()

#plt.plot(np.arange(0, len(trainLoss)), trainLoss, label="train_loss")
#plt.plot(np.arange(0, len(valLoss)), valLoss, label="val_loss")
#plt.plot(np.arange(0, len(trainRank1)), trainRank1, label="train_acc")
#plt.plot(np.arange(0, len(valRank1)), valRank1,	label="val_acc")
plt.title("{}: loss & accuracy on {}".format(args["network"], args["dataset"]))
plt.xlabel("Epoch #")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
#ax1.legend(loc="upper right")
plt.savefig('loss_acc.png')
#plt.show()
