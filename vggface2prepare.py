# import the necessary packages
import numpy as np
import os

class VGGFace2Prepare:
	def __init__(self, config):
		# store the configuration object
		self.config = config

		# build the label mappings and validation blacklist
		self.labelMappings = self.buildClassLabels()
	#	self.valBlacklist = self.buildBlackist()

	def buildClassLabels(self):
		# load the contents of the file that maps the WordNet IDs
		# to integers, then initialize the label mappings dictionary
		rows = open(self.config.WORD_IDS).read().strip().split("\n")
		labelMappings = {}

		# loop over the labels
		for row in rows:
			# split the row into the WordNet ID, label integer, and
			# human readable label
			(wordID, label) = row.split(",")[:2]

			# update the label mappings dictionary using the word ID
			# as the key and the label as the value, subtracting `1`
			# from the label since MATLAB is one-indexed while Python
			# is zero-indexed
                        #labelMappings[wordID] = int(label) - 1
			labelMappings[wordID] = label

		# return the label mappings dictionary
		return labelMappings

	#def buildBlackist(self):
		# load the list of blacklisted image IDs and convert them to
		# a set
	#	rows = open(self.config.VAL_BLACKLIST).read()
	#	rows = set(rows.strip().split("\n"))

		# return the blacklisted image IDs
	#	return rows

	def buildTrainingSet(self):
		# load the contents of the training input file that lists
		# the partial image ID and image number, then initialize
		# the list of image paths and class labels
		rows = open(self.config.TRAIN_LIST).read().strip()
		rows = rows.split("\n")
		paths = []
		labels = []

		# loop over the rows in the input training file
		for row in rows:
			# break the row into the the partial path and image
			# number (the image number is sequential and is
			# essentially useless to us)
			#partialPath = row.strip().split(" ")
			partialPath = row

			# construct the full path to the training image, then
			# grab the word ID from the path and use it to determine
			# the integer class label
			#path = os.path.sep.join([self.config.IMAGES_PATH, partialPath])
			path = partialPath
			wordID = partialPath.split("/")[0]
			label = self.labelMappings[wordID]

			# update the respective paths and label lists
			paths.append(path)
			labels.append(label)


		# return a tuple of image paths and associated integer class
		# labels
		return (np.array(paths), np.array(labels))

	def buildTestingSet(self):
		# initialize the list of image paths and class labels
		rows = open(self.config.TEST_LIST).read().strip()
		rows = rows.split("\n")
		paths = []
		labels = []

		# loop over the rows in the input training file
		for row in rows:
			# break the row into the the partial path and image
			# number (the image number is sequential and is
			# essentially useless to us)
			#partialPath = row.strip().split(" ")
			partialPath = row

			# construct the full path to the training image, then
			# grab the word ID from the path and use it to determine
			# the integer class label
			#path = os.path.sep.join([self.config.IMAGES_PATH_TEST, partialPath])
			path = partialPath
			wordID = partialPath.split("/")[0]
			label = self.labelMappings[wordID]

			# update the respective paths and label lists
			paths.append(path)
			labels.append(label)
		# return a tuple of image paths and associated integer class
		# labels
		return (np.array(paths), np.array(labels))
