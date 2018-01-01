# USAGE
# python test_prediction.py --checkpoints ~/dcnn/datasets/vggface2/checkpoints/resnet101_checkpoints/ --prefix resnet101 --epoch 19 -s 10

# import OpenCV before mxnet to avoid a segmentation fault
import cv2

# import the necessary packages
from config import vggface2_config as config
from neuralnetwork.preprocessing import ImageToArrayPreprocessor
from neuralnetwork.preprocessing import SimplePreprocessor
from neuralnetwork.preprocessing import MeanPreprocessor
from neuralnetwork.preprocessing import CropPreprocessor
#from neuralnetwork.utils import FaceHelper
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils import paths
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import json
import dlib
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to the checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-e", "--epoch", type=int, required=True,
	help="epoch # to load")
ap.add_argument("-s", "--sample-size", type=int, default=10,
	help="sample # of images to load")
args = vars(ap.parse_args())

# load the label encoders and mean files
#print("[INFO] loading label encoders and mean files...")
#faceLE = pickle.loads(open(config.FACE_LABEL_ENCODER, "rb").read())
faceMeans = json.loads(open(config.DATASET_MEAN).read())

# load the label encoder, followed by the testing dataset file,
# then sample the testing set
#rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = open(config.TRAIN_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows, size=args["sample_size"])

# load our pre-trained model
print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],
	args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath,
	args["epoch"])

# compile the model
model = mx.model.FeedForward(
	ctx=[mx.gpu(0)],
	symbol=model.symbol,
	arg_params=model.arg_params,
	aux_params=model.aux_params)

# initialize the image pre-processors
sp = SimplePreprocessor(width=256, height=256,
	inter=cv2.INTER_CUBIC)
cp = CropPreprocessor(width=227, height=227, horiz=True)
faceMP = MeanPreprocessor(faceMeans["R"], faceMeans["G"],
	faceMeans["B"])
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

# initialize dlib's face detector (HOG-based), then create the
# the facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize the list of image paths as just a single image
#imagePaths = [args["image"]]

# if the input path is actually a directory, then list all image
# paths in the directory
#if os.path.isdir(args["image"]):
#	imagePaths = sorted(list(paths.list_files(args["image"])))

# loop over the image paths
#for imagePath in imagePaths:
# loop over the testing images
for row in rows:
	# grab the target class label and the image path from the row
	(target, imagePath) = row.split("\t")[1:]
	target = int(target)
	# load the image from disk, resize it, and convert it to
	# grayscale
	print("[INFO] processing {}".format(imagePath))
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# align the face
		shape = predictor(gray, rect)
		face = fa.align(image, gray, rect)

		# resize the face to a fixed size, then extract 10-crop
		# patches from it
		face = sp.preprocess(face)
		patches = cp.preprocess(face)

		# allocate memory for the face patches
		facePatches = np.zeros((patches.shape[0], 3, 227, 227),
			dtype="float")

		# loop over the patches
		for j in np.arange(0, patches.shape[0]):
			# perform mean subtraction on the patch
			facePatch = faceMP.preprocess(patches[j])
			facePatch = iap.preprocess(facePatch)

			# update the respective patches lists
			facePatches[j] = facePatch

		# make predictions on faces based on the extracted
		# patches
		facePreds = model.predict(facePatches)[0]

		# compute the average for each class label based on the
		# predictions for the patches
		#facePreds = facePreds.mean(axis=0)
		
		facePreds5 = np.argsort(facePreds)
		
		# show the true class label
		print("[INFO] actual={}".format(target))
		
		print("[INFO] facePredsvalue={}".format(facePreds5))
		
		#print("[INFO] facePredsvalue={}".format(facePreds5))
		
		#for (i, pred) in enumerate(facePreds5):	
			# show the predicted class label
			#print("[INFO] predicted={}".format(pred))
			#print("[INFO] ivalue={}".format(i))
			
			

		# visualize the face predictions
		#faceCanvas = FaceHelper.visualizeFace(facePreds,
		#	faceLE)

		# draw the bounding box around the face
		#clone = image.copy()
		#(x, y, w, h) = face_utils.rect_to_bb(rect)
		#cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the output image
		#cv2.imshow("Input", clone)
		#cv2.imshow("Face", face)
		#cv2.imshow("Face Probabilities", faceCanvas)
		#cv2.waitKey(0)