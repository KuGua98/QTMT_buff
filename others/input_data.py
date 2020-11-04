from others import file_reader as fr
import os
import numpy as np
import tensorflow as tf

train_data_dir = 'E:/QTMT/train_samples/'
valid_data_dir = 'E:/QTMT/valid_samples/'

# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
NAME_LIST = ['64x64','32x32','16x16','32x16','8x8','32x8','16x8','8x4','32x4','16x4']
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]
WIDTH_LIST = [64, 32, 16, 32, 8, 32, 16, 8, 32, 16]
HEIGHT_LIST = [64, 32, 16, 16, 8, 8, 8, 4, 4, 4]


NUM_CHANNELS = 1

TRAINSET_MAXSIZE = 1000000
TRAINSET_READSIZE = 800000

VALIDSET_MAXSIZE = 200000
VALIDSET_READSIZE = 100000

TRAIN_FILE_READER = []
TRAINSET = []

VALID_FILE_READER = []
VALIDSET = []


def get_size(file_name):
    global SAMPLE_LENGTH, IMAGE_LENGTH, WIDTH, HEIGHT
    for i in range(len(NAME_LIST)):
        if NAME_LIST[i] == file_name:
            SAMPLE_LENGTH = SAMPLE_LENGTH_LIST[i]
            IMAGE_LENGTH = IMAGES_LENGTH_LIST[i]
            WIDTH = WIDTH_LIST[i]
            HEIGHT = HEIGHT_LIST[i]

def get_train_valid_sets(file_name):
    global TRAINSET, VALIDSET
    TRAINSET = file_name+'_train'+'.dat'
    VALIDSET = file_name+'_valid'+'.dat'

def getFileSize(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)

def get_data_set(file_reader, read_bytes, is_loop=True):
    data = file_reader.read_data(read_bytes, isloop=is_loop)
    data_amount = len(data)
    assert data_amount % (SAMPLE_LENGTH) == 0
    num_samples = int(data_amount / (SAMPLE_LENGTH))

    data = data.reshape(num_samples, SAMPLE_LENGTH)

    images = data[:, 0:IMAGE_LENGTH]
    images = np.reshape(images, [-1, WIDTH, HEIGHT, NUM_CHANNELS])

    rd_cost = data[:,IMAGE_LENGTH:IMAGE_LENGTH+6]
    qp = data[:,IMAGE_LENGTH+6]
    rd_cost_min = data[:,IMAGE_LENGTH+7]
    partition_label = data[:,IMAGE_LENGTH+8]
    # label转化为one - hot形式
    if IMAGE_LENGTH == 4096:
        label = tf.one_hot(indices=partition_label, depth=2)
    else:
        label = tf.one_hot(indices=partition_label, depth=6)

    return DataSet(images, rd_cost, qp, rd_cost_min, label)

class DataSet(object):
    def __init__(self, images, rd_cost, qps, rd_cost_min, partition_label):
        # assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._partition_label = partition_label
        self._qps = qps
        self._rd_cost = rd_cost
        self._rd_cost_min = rd_cost_min
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._partition_label

    @property
    def qps(self):
        return self._qps

    @property
    def rd_cost(self):
        return self._rd_cost

    @property
    def rd_cost_min(self):
        return self._rd_cost_min

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._partition_label[start:end], self._qps[start:end],\
               self._rd_cost[start:end], self._rd_cost_min[start:end]


def read_data_set(file_name):
    global TRAIN_FILE_READER, VALID_FILE_READER
    class DataSets(object):
        pass

    get_size(file_name)

    data_sets = DataSets()
    data_sets.train = []
    data_sets.valid = []

    get_train_valid_sets(file_name)

    TRAIN_FILE_READER = fr.FileReader()
    TRAIN_FILE_READER.initialize(os.path.join(train_data_dir, TRAINSET), TRAINSET_MAXSIZE * SAMPLE_LENGTH)

    VALID_FILE_READER = fr.FileReader()
    VALID_FILE_READER.initialize(os.path.join(valid_data_dir, VALIDSET), VALIDSET_MAXSIZE * SAMPLE_LENGTH)

    data_sets.train = get_data_set(TRAIN_FILE_READER, TRAINSET_READSIZE * SAMPLE_LENGTH)
    data_sets.valid = get_data_set(VALID_FILE_READER, VALIDSET_READSIZE * SAMPLE_LENGTH)

    return data_sets