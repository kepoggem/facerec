# import the necessary packages
from os import path

# define the base path to where the ImageNet dataset
# devkit are stored on disk)
BASE_PATH = "/home/kepoggem/dcnn/datasets/vggface2"

# based on the base path, derive the images base path, image sets
# path, and devkit path
IMAGES_PATH = path.sep.join([BASE_PATH, "train"])
IMAGES_PATH_TEST = path.sep.join([BASE_PATH, "test"])
#IMAGE_SETS_PATH = path.sep.join([BASE_PATH, "ImageSets/CLS-LOC/"])
DEVKIT_PATH = path.sep.join([BASE_PATH, "meta"])

# define the path that maps the 1,000 possible WordNet IDs to the
# class label integers
WORD_IDS = path.sep.join([DEVKIT_PATH, "identity_meta.csv"])

# define the paths to the training file that maps the (partial)
# image filename to integer class label
TRAIN_LIST = path.sep.join([DEVKIT_PATH, "train_list.txt"])

# define the paths to to the validation filenames along with the
# file that contains the ground-truth validation labels
TEST_LIST = path.sep.join([DEVKIT_PATH, "test_list.txt"])
#VAL_LABELS = path.sep.join([DEVKIT_PATH,
#	"ILSVRC2015_clsloc_validation_ground_truth.txt"])

# define the path to the validation files that are blacklisted
#VAL_BLACKLIST = path.sep.join([DEVKIT_PATH,
#	"ILSVRC2015_clsloc_validation_blacklist.txt"])

# since we do not have access to the testing data we need to
# take a number of images from the training data and use it instead
#NUM_CLASSES = 1000
NUM_CLASSES = 1057
#NUM_TEST_IMAGES = 10 * NUM_CLASSES
NUM_TEST_IMAGES = 15307
NUM_VAL_IMAGES = 0.20

# define the path to the output training, validation, and testing
# lists
MX_OUTPUT = BASE_PATH
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

# define the path to the output training, validation, and testing
# image records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

# define the path to the dataset mean
DATASET_MEAN = "/home/kepoggem/dcnn/datasets/vggface2/output/vggface2_mean.json"

# define the batch size and number of devices used for training
BATCH_SIZE = 32
NUM_DEVICES = 1
