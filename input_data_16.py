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
# TRAIN_SAMPLE_PATH = "D:/QTMT/TRAIN_SAMPLE/"
VALID_SAMPLE_PATH = "D:/QTMT/CU_SAMPLE_VALID/"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 或者直接按固定的比例分配。以下代码会占用所有可使用GPU的40%显存。
config.gpu_options.per_process_gpu_memory_fraction = 0.4

def gen_train1():
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    _, size_dataset_all = gd.get_train_data_size(CU_NAME1)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_train"
    cu_file = h5py.File(TRAIN_SAMPLE_PATH+CU_NAME1+MODE_NAME + '.h5', 'r')
    cu_dataset = cu_file[CU_NAME1]

    index=0
    while True:
        data_buff = cu_dataset[index]
        image = data_buff[:IMAGES_LENGTH1]
        label = data_buff[IMAGES_LENGTH1 + 8]
        qp = data_buff[IMAGES_LENGTH1 + 6]

        image = image.reshape([CU_WIDTH1, CU_HEIGHT1, 1])
        # label = tf.one_hot(indices=label, depth=LABEL_LENGTH)
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH1).eval(session=sess)
        qp = qp.reshape([1])

        yield (image, label, qp)
        index += 1
        if index == size_dataset_all-1:
            index = 0

def gen_valid1():
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    _, size_dataset_all = gd.get_valid_data_size(CU_NAME1)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_valid"
    cu_file = h5py.File(VALID_SAMPLE_PATH+CU_NAME1 + MODE_NAME + '.h5', 'r')
    cu_dataset = cu_file[CU_NAME1]

    index = 0
    while True:
        data_buff = cu_dataset[index]
        image = data_buff[:IMAGES_LENGTH1]
        label = data_buff[IMAGES_LENGTH1 + 8]
        qp = data_buff[IMAGES_LENGTH1 + 6]

        image = image.reshape([CU_WIDTH1, CU_HEIGHT1, 1])
        # label = tf.one_hot(indices=label, depth=LABEL_LENGTH)
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH1).eval(session=sess)
        qp = qp.reshape([1])

        yield (image, label, qp)
        index += 1
        if index == size_dataset_all - 1:
            index = 0


def get_train_dataset1(cu_width, cu_height, label_length, images_length):
    global CU_NAME1, CU_WIDTH1, CU_HEIGHT1, LABEL_LENGTH1, IMAGES_LENGTH1
    CU_NAME1 = str(cu_width) + 'x' + str(cu_height)
    CU_WIDTH1 = cu_width
    CU_HEIGHT1 = cu_height
    LABEL_LENGTH1 = label_length
    IMAGES_LENGTH1 = images_length

    data = tf.data.Dataset.from_generator(gen_train1, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width,cu_height,1]), tf.TensorShape([label_length]), tf.TensorShape([1])))
    data = data.shuffle(TRAINSET_READSIZE)
    data = data.batch(MINI_BATCH_SIZE)
    data = data.repeat()
    data = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch = data.get_next()

    return  images_batch, label_batch, qp_batch

def get_valid_dataset1(cu_width, cu_height, label_length):
    # CU_NAME = str(cu_width) + 'x' + str(cu_height)
    data = tf.data.Dataset.from_generator(gen_valid1, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width, cu_height, 1]), tf.TensorShape([label_length]), tf.TensorShape([1])))
    data = data.shuffle(VALIDSET_READSIZE)
    data = data.batch(MINI_BATCH_SIZE)
    data = data.repeat()
    data = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch = data.get_next()

    return images_batch, label_batch, qp_batch




def gen_train2():
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    _, size_dataset_all = gd.get_train_data_size(CU_NAME2)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_train"
    cu_file = h5py.File(TRAIN_SAMPLE_PATH+CU_NAME2+MODE_NAME + '.h5', 'r')
    cu_dataset = cu_file[CU_NAME2]

    index=0
    while True:
        data_buff = cu_dataset[index]
        image = data_buff[:IMAGES_LENGTH2]
        label = data_buff[IMAGES_LENGTH2 + 8]
        qp = data_buff[IMAGES_LENGTH2 + 6]

        image = image.reshape([CU_WIDTH2, CU_HEIGHT2, 1])
        # label = tf.one_hot(indices=label, depth=LABEL_LENGTH)
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH2).eval(session=sess)
        qp = qp.reshape([1])

        yield (image, label, qp)
        index += 1
        if index == size_dataset_all-1:
            index = 0

def gen_valid2():
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    _, size_dataset_all = gd.get_valid_data_size(CU_NAME2)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_valid"
    cu_file = h5py.File(VALID_SAMPLE_PATH+CU_NAME2 + MODE_NAME + '.h5', 'r')
    cu_dataset = cu_file[CU_NAME2]

    index = 0
    while True:
        data_buff = cu_dataset[index]
        image = data_buff[:IMAGES_LENGTH2]
        label = data_buff[IMAGES_LENGTH2 + 8]
        qp = data_buff[IMAGES_LENGTH2 + 6]

        image = image.reshape([CU_WIDTH2, CU_HEIGHT2, 1])
        # label = tf.one_hot(indices=label, depth=LABEL_LENGTH)
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH2).eval(session=sess)
        qp = qp.reshape([1])

        yield (image, label, qp)
        index += 1
        if index == size_dataset_all - 1:
            index = 0


def get_train_dataset2(cu_width, cu_height, label_length, images_length):
    global CU_NAME2, CU_WIDTH2, CU_HEIGHT2, LABEL_LENGTH2, IMAGES_LENGTH2
    CU_NAME2 = str(cu_width) + 'x' + str(cu_height)
    CU_WIDTH2 = cu_width
    CU_HEIGHT2 = cu_height
    LABEL_LENGTH2 = label_length
    IMAGES_LENGTH2 = images_length

    data = tf.data.Dataset.from_generator(gen_train2, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width,cu_height,1]), tf.TensorShape([label_length]), tf.TensorShape([1])))
    data = data.shuffle(TRAINSET_READSIZE)
    data = data.batch(MINI_BATCH_SIZE)
    data = data.repeat()
    data = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch = data.get_next()

    return  images_batch, label_batch, qp_batch

def get_valid_dataset2(cu_width, cu_height, label_length):

    # CU_NAME = str(cu_width) + 'x' + str(cu_height)
    data = tf.data.Dataset.from_generator(gen_valid2, (tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width, cu_height, 1]), tf.TensorShape([label_length]), tf.TensorShape([1])))
    data = data.shuffle(VALIDSET_READSIZE)
    data = data.batch(MINI_BATCH_SIZE)
    data = data.repeat()
    data = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch = data.get_next()

    return images_batch, label_batch, qp_batch