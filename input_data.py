import sys
sys.path.append("..")
import tensorflow as tf
import h5py
from extract_data import get_details as gd
import data_info as di
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAINSET_READSIZE = 1000 # shuffle size
VALIDSET_READSIZE = 1000

MINI_BATCH_SIZE = 32

# INDEX_LIST_TRAIN = list(range(0,164))
# INDEX_LIST_VALID = list(range(164,190))

TRAIN_SAMPLE_PATH = "D:/QTMT/CU_SAMPLE_TRAIN/"
VALID_SAMPLE_PATH = "D:/QTMT/CU_SAMPLE_VALID/"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 或者直接按固定的比例分配。以下代码会占用所有可使用GPU的40%显存。
config.gpu_options.per_process_gpu_memory_fraction = 0.4

def gen_train():
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    _, size_dataset_all = gd.get_train_data_size(CU_NAME)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_train"
    cu_file = h5py.File(TRAIN_SAMPLE_PATH+CU_NAME+MODE_NAME + '.h5', 'r')
    cu_dataset = cu_file[CU_NAME]

    index=0
    while True:
        data_buff = cu_dataset[index]
        image = data_buff[:IMAGES_LENGTH]
        label = data_buff[IMAGES_LENGTH + 8]
        qp = data_buff[IMAGES_LENGTH + 6]
        RDcost = data_buff[IMAGES_LENGTH: IMAGES_LENGTH+2]
        min_RDcost = data_buff[IMAGES_LENGTH + 7]

        image = image.reshape([CU_WIDTH, CU_HEIGHT, 1])
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH).eval(session=sess)
        qp = qp.reshape([1])
        min_RDcost = min_RDcost.reshape([1])
        RDcost = RDcost.reshape([LABEL_LENGTH])

        yield (image, label, qp, min_RDcost, RDcost)
        index += 1
        if index == size_dataset_all:
            break
        # if index == size_dataset_all-1:
        #     index = 0

def gen_valid():
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    _, size_dataset_all = gd.get_valid_data_size(CU_NAME)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_valid"
    cu_file = h5py.File(VALID_SAMPLE_PATH+CU_NAME + MODE_NAME + '.h5', 'r')
    cu_dataset = cu_file[CU_NAME]

    index = 0
    while True:
        data_buff = cu_dataset[index]
        image = data_buff[:IMAGES_LENGTH]
        label = data_buff[IMAGES_LENGTH + 8]
        qp = data_buff[IMAGES_LENGTH + 6]
        RDcost = data_buff[IMAGES_LENGTH: IMAGES_LENGTH + 2]
        min_RDcost = data_buff[IMAGES_LENGTH + 7]

        image = image.reshape([CU_WIDTH, CU_HEIGHT, 1])
        # label = tf.one_hot(indices=label, depth=LABEL_LENGTH)
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH).eval(session=sess)
        qp = qp.reshape([1])
        min_RDcost = min_RDcost.reshape([1])
        RDcost = RDcost.reshape([LABEL_LENGTH])

        yield (image, label, qp, min_RDcost, RDcost)
        index += 1
        if index == size_dataset_all:
            break
        # if index == size_dataset_all - 1:
        #     index = 0


def get_train_dataset(cu_width, cu_height, label_length, images_length):
    global CU_NAME, CU_WIDTH, CU_HEIGHT, LABEL_LENGTH, IMAGES_LENGTH
    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    CU_WIDTH = cu_width
    CU_HEIGHT = cu_height
    LABEL_LENGTH = label_length
    IMAGES_LENGTH = images_length

    data = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width, cu_height, 1]), tf.TensorShape([label_length]), tf.TensorShape([1]), tf.TensorShape([1]), tf.TensorShape([label_length])))
    data = data.shuffle(TRAINSET_READSIZE)
    data = data.repeat()
    data = data.batch(MINI_BATCH_SIZE)
    iterator = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch = iterator.get_next()

    return images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch


    # data_buff = cu_dataset[:size_dataset_all]
    #
    # images_train = data_buff[0:size_dataset_all, :IMAGES_LENGTH]
    # label_train = data_buff[0:size_dataset_all, IMAGES_LENGTH+8]
    # qp_train = data_buff[0:size_dataset_all, IMAGES_LENGTH+6]
    # assert images_train.shape[0] == label_train.shape[0]
    # assert images_train.shape[0] == qp_train.shape[0]
    #
    # sess = tf.Session()
    # images_train = images_train.reshape([-1, CU_WIDTH, CU_HEIGHT, 1])
    # label_train = tf.one_hot(indices=label_train, depth=LABEL_LENGTH).eval(session=sess)
    # qp_train = qp_train.reshape([-1, 1])
    #
    # return images_train, label_train, qp_train



def get_valid_dataset(cu_width, cu_height, label_length):

    # CU_NAME = str(cu_width) + 'x' + str(cu_height)
    data = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width, cu_height, 1]), tf.TensorShape([label_length]), tf.TensorShape([1]), tf.TensorShape([1]), tf.TensorShape([label_length])))
    data = data.shuffle(VALIDSET_READSIZE)
    data = data.repeat()
    data = data.batch(MINI_BATCH_SIZE)
    iterator = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch = iterator.get_next()

    return images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch
