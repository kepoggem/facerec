# USAGE
# python train.py --checkpoints ~/dcnn/datasets/vggface2/checkpoints/vggface_checkpoints --prefix vggface

# import the necessary packages
from config import vggface2_config as config
from neuralnetwork.nn.mxconv import MxVGGNet
from vggface2prepare import VGGFace2Prepare
from neuralnetwork.mxcallbacks import one_off_callback
import mxnet as mx
import argparse
import logging
import pickle
import json
import os


##################
class DataBath(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label

class DataIter(mx.io.DataIter):
    def __init__(self, images, batch_size, height, width, process_num):
        assert process_num <= 40
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.conut = len(images)
        self.height = height
        self.width = width
        self.images = images
        self.cursor = -self.batch_size
        self.provide_data = [("positive", (self.batch_size, 3, height, width)),
                             ("negative", (self.batch_size, 3, height, width)),
                             ("one", (self.batch_size, ))]
        self.provide_label = [("anchor", (self.batch_size, 3, height, width))]
        self.queue = multiprocessing.Queue(maxsize=4)
        self.started = True
        self.processes = [multiprocessing.Process(target=self.write) for i in range(process_num)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def augment(self, mat):
        # bright = random.randint(60, 100)/100.0
        # mat = cv2.convertScaleAbs(mat, None, bright, 0)
        # mat = cv2.GaussianBlur(mat, (3, 3), 0, 0, borderType=cv2.BORDER_REPLICATE)
        rows, cols, _ = mat.shape
        # print rows, cols
        x_scale = random.randint(-12, 12) / 100.0
        y_scale = random.randint(-12, 12) / 100.0
        # x_scale = 0.1
        # y_scale = -0.1
        x_resize_scale = cols / (cols + abs(x_scale) * rows)
        y_resize_scale = rows / (rows + abs(y_scale) * cols)
        if x_scale >= 0:
            if y_scale >= 0:
                affine_matrix = np.float32([[x_resize_scale, x_resize_scale * x_scale, 0],
                                            [y_resize_scale * y_scale, y_resize_scale, 0]])
            else:
                affine_matrix = np.float32([[x_resize_scale, x_resize_scale * x_scale, 0],
                                            [y_resize_scale * y_scale, y_resize_scale,
                                             y_resize_scale * abs(y_scale) * cols]])
        else:
            if y_scale >= 0:
                affine_matrix = np.float32(
                    [[x_resize_scale, x_resize_scale * x_scale, x_resize_scale * abs(x_scale) * rows],
                     [y_resize_scale * y_scale, y_resize_scale, 0]])
            else:
                affine_matrix = np.float32(
                    [[x_resize_scale, x_resize_scale * x_scale, x_resize_scale * abs(x_scale) * rows],
                     [y_resize_scale * y_scale, y_resize_scale, y_resize_scale * abs(y_scale) * cols]])
        affine_mat = cv2.warpAffine(mat, affine_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        return affine_mat

    def generate_batch(self):
        ret = []
        while len(ret) < self.batch_size:
            #a_idx, n_idx = random.sample(range(self.conut), 2)
            batch1 = trainIter.next() # first batch.
			a_idx = batch1.data[0]
			batch2 = trainIter.next() # first batch.
			n_idx = batch2.data[0]

            if a_idx == n_idx:
                continue
            a_mat = cv2.imread(self.images[a_idx])
            a_mat = cv2.resize(a_mat, (self.height, self.width))
            p_mat = self.augment(a_mat)
            p_mat = cv2.resize(p_mat, (self.height, self.width))
            n_mat = cv2.imread(self.images[n_idx])
            n_mat = cv2.resize(n_mat, (self.height, self.width))
            threshold = 250
            if np.mean(a_mat) > threshold or np.mean(p_mat) > threshold or np.mean(n_mat) > threshold:
                continue
            ret.append((a_mat, p_mat, n_mat))
        return ret

    def write(self):
        while True:
            if not self.started:
                break
            batch = self.generate_batch()
            a_batch = [x[0].transpose(2, 0, 1) for x in batch]
            p_batch = [x[1].transpose(2, 0, 1) for x in batch]
            n_batch = [x[2].transpose(2, 0, 1) for x in batch]
            one_batch = np.ones(self.batch_size)
            data_all = [mx.nd.array(p_batch),
                        mx.nd.array(n_batch),
                        mx.nd.array(one_batch)]
            label_all = [mx.nd.array(a_batch)]
            data_batch = DataBath(data_all, label_all)
            self.queue.put(data_batch)

    def __del__(self):
        self.started = False
        for process in self.processes:
            process.join()
            while not self.queue.empty():
                self.queue.get(block=False)

    def next(self):
        if self.queue.empty():
            logging.debug("waitting for data......")
        if self.iter_next():
            return self.queue.get(block=True)
        else:
            raise StopIteration

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.conut

    def reset(self):
        self.cursor = -self.batch_size


##########
#parser = argparse.ArgumentParser(description="Image Search Using CNN")
    #parser.add_argument("--batch_size", type=int, default=64)
    #parser.add_argument("--gpus", type=int, default=0)
 #   parser.add_argument("--process_num", type=int, default=4)
    parser.add_argument("--root", type=str, default="")
  #  args = parser.parse_args()
   # batch_size = args.batch_size
    #dev = args.gpus
    #network = get_network(batch_size=batch_size)
    # symbol, arg_params, aux_params = mx.model.load_checkpoint("resnet-18", 0)

    # shape = {"anchor": (batch_size, 3, 224, 224),
    #          "positive": (batch_size, 3, 224, 224),
    #          "negative": (batch_size, 3, 224, 224),
    #          "one": (batch_size, )}
    # mx.visualization.plot_network(network, shape=shape).render("ir-resnet", cleanup=True)
#################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True,
	help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
add_argument("--root", type=str, default="")
args = vars(ap.parse_args())

# set the logging level and output file
logging.basicConfig(level=logging.DEBUG,
	filename="training_vgg19_{}.log".format(args["start_epoch"]),
	filemode="w")

# determine the batch and load the mean pixel values
#batchSize = config.BATCH_SIZE * config.NUM_DEVICES
batchSize = 64
batch_size = batchSize
means = json.loads(open(config.DATASET_MEAN).read())

# construct the training image iterator
trainIter = mx.io.ImageRecordIter(
	path_imgrec=config.TRAIN_MX_REC,
	data_shape=(3, 227, 227),
	batch_size=1,
	rand_crop=True,
	rand_mirror=True,
	rotate=7,
	mean_r=means["R"],
	mean_g=means["G"],
	mean_b=means["B"],
	preprocess_threads=config.NUM_DEVICES * 2)



# construct the validation image iterator
valIter = mx.io.ImageRecordIter(
	path_imgrec=config.VAL_MX_REC,
	data_shape=(3, 227, 227),
	batch_size=batchSize,
	mean_r=means["R"],
	mean_g=means["G"],
	mean_b=means["B"])

# initialize the optimizer
opt = mx.optimizer.SGD(learning_rate=1e-2, momentum=0.9, wd=0.0001,
	rescale_grad=1.0 / batchSize)

# construct the checkpoints path, initialize the model argument and
# auxiliary parameters
checkpointsPath = os.path.sep.join([args["checkpoints"],
	args["prefix"]])
argParams = None
auxParams = None

# if there is no specific model starting epoch supplied, then
# initialize the network
if args["start_epoch"] <= 0:
	# build the LeNet architecture
	print("[INFO] building network...")
	model = MxVGGNetEmbeddings.build(config.NUM_CLASSES)

# otherwise, a specific checkpoint was supplied
else:
	# load the checkpoint from disk
	print("[INFO] loading epoch {}...".format(args["start_epoch"]))
	(model, argParams, auxParams) = mx.model.load_checkpoint(
		checkpointsPath, args["start_epoch"])

################
    images = []
    root_dir = args.root
    for root, dirnames, filenames in os.walk(root_dir):
        for img in fnmatch.filter(filenames, "*.jpg"):
            images.append(os.path.abspath(os.path.join(root, img)))

    train_set = DataIter(images=images, batch_size=batch_size, height=227, width=227, process_num=args.process_num)
################

# compile the model
model = mx.model.FeedForward(
	ctx=[mx.gpu(0), mx.gpu(1)],
	symbol=model,
	initializer=mx.initializer.Xavier(),
	arg_params=argParams,
	aux_params=auxParams,
	optimizer=opt,
	num_epoch=110,
	begin_epoch=args["start_epoch"])

# initialize the callbacks and evaluation metrics
batchEndCBs = [mx.callback.Speedometer(batchSize, 10)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.CrossEntropy()]

# train the network
print("[INFO] training network...")
model.fit(
	X=trainIter,
	eval_data=valIter,
	eval_metric=metrics,
	batch_end_callback=batchEndCBs,
	epoch_end_callback=epochEndCBs)
