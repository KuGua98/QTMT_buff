import sys
sys.path.append("..")
import tensorflow as tf
import h5py
from extract_data import get_details as gd
import data_info as di
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAINSET_READSIZE = 1000 # shuffle size
VALIDSET_READSIZE = 1000

MINI_BATCH_SIZE = 32

TRAIN_SAMPLE_PATH = "D:/QTMT/CU_SAMPLE_TRAIN/"
VALID_SAMPLE_PATH = "D:/QTMT/CU_SAMPLE_VALID/"

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
        RDcost = data_buff[IMAGES_LENGTH: IMAGES_LENGTH + 6]
        min_RDcost = data_buff[IMAGES_LENGTH + 7]

        if CU_NAME == '32x16':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0 or RDcost[4] == 0 or RDcost[5] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0], RDcost[2], RDcost[3], RDcost[4], RDcost[5]])

            assert label != 1
            if label > 1:
                label -= 1
        elif CU_NAME == '8x8':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0], RDcost[2], RDcost[3]])

            assert label == 0 or label == 2 or label == 3
            if label > 1:
                label -= 1
        elif CU_NAME == '16x8':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0 or RDcost[4] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0], RDcost[2], RDcost[3], RDcost[4]])

            assert label != 1 and label != 5
            if label > 1:
                label -= 1
        elif CU_NAME == '32x8':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0 or RDcost[5] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0], RDcost[2], RDcost[3], RDcost[5]])

            assert label != 1 and label != 4
            if label == 2 or label == 3:
                label -= 1
            elif label == 5:
                label -= 2
        elif CU_NAME == '8X4':

            if RDcost[0] == 0 or RDcost[3] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0], RDcost[3]])

            assert label == 0 or label == 3
            if label == 3:
                label -= 2
        elif CU_NAME == '16X4' or CU_NAME == '32X4':

            if RDcost[0] == 0 or RDcost[3] == 0 or RDcost[5] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0], RDcost[3], RDcost[5]])

            assert label == 0 or label == 3 or label == 5
            if label == 3:
                label -= 2
            elif label == 5:
                label -= 3

        image = image.reshape([CU_WIDTH, CU_HEIGHT, 1])
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH).eval(session=sess)
        qp = qp.reshape([1])
        min_RDcost = min_RDcost.reshape([1])
        RDcost_save = RDcost_save.reshape([LABEL_LENGTH])

        yield (image, label, qp, min_RDcost, RDcost_save)
        index += 1
        if index == size_dataset_all:
            break


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
        RDcost = data_buff[IMAGES_LENGTH: IMAGES_LENGTH+6]
        min_RDcost = data_buff[IMAGES_LENGTH + 7]

        if CU_NAME == '32x16':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0 or RDcost[4] == 0 or RDcost[5] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0],RDcost[2],RDcost[3],RDcost[4],RDcost[5]])

            assert label != 1
            if label > 1:
                label -= 1
        elif CU_NAME == '8x8':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0],RDcost[2],RDcost[3]])

            assert label == 0 or label == 2 or label == 3
            if label > 1:
                label -= 1
        elif CU_NAME == '16x8':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0 or RDcost[4] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0],RDcost[2],RDcost[3],RDcost[4]])

            assert label != 1 and label != 5
            if label > 1:
                label -= 1
        elif CU_NAME == '32x8':

            if RDcost[0] == 0 or RDcost[2] == 0 or RDcost[3] == 0 or RDcost[5] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0],RDcost[2],RDcost[3],RDcost[5]])

            assert label != 1 and label != 4
            if label == 2 or label == 3:
                label -= 1
            elif label == 5:
                label -= 2
        elif CU_NAME == '8X4':

            if RDcost[0] == 0 or RDcost[3] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0],RDcost[3]])

            assert label == 0 or label == 3
            if label == 3:
                label -= 2
        elif CU_NAME == '16X4' or CU_NAME == '32X4':

            if RDcost[0] == 0 or RDcost[3] == 0 or RDcost[5] == 0:
                index += 1
                continue
            RDcost_save = np.array([RDcost[0],RDcost[3],RDcost[5]])

            assert label == 0 or label == 3 or label == 5
            if label == 3:
                label -= 2
            elif label == 5:
                label -= 3

        image = image.reshape([CU_WIDTH, CU_HEIGHT, 1])
        # label = tf.one_hot(indices=label, depth=LABEL_LENGTH)
        sess = tf.Session()
        label = tf.one_hot(indices=label, depth=LABEL_LENGTH).eval(session=sess)
        qp = qp.reshape([1])
        min_RDcost = min_RDcost.reshape([1])
        RDcost_save = RDcost_save.reshape([LABEL_LENGTH])

        yield (image, label, qp, min_RDcost, RDcost_save)
        index += 1
        if index == size_dataset_all:
            break


def get_train_dataset(cu_width, cu_height, label_length, images_length):
    global CU_NAME, CU_WIDTH, CU_HEIGHT, LABEL_LENGTH, IMAGES_LENGTH
    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    CU_WIDTH = cu_width
    CU_HEIGHT = cu_height
    LABEL_LENGTH = label_length
    IMAGES_LENGTH = images_length

    data = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width,cu_height,1]), tf.TensorShape([label_length]), tf.TensorShape([1]), tf.TensorShape([1]), tf.TensorShape([label_length])))
    data = data.shuffle(TRAINSET_READSIZE)
    data = data.repeat()
    data = data.batch(MINI_BATCH_SIZE)
    iterator = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch = iterator.get_next()

    return  images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch


def get_valid_dataset(cu_width, cu_height, label_length):

    # CU_NAME = str(cu_width) + 'x' + str(cu_height)
    data = tf.data.Dataset.from_generator(gen_valid, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), (tf.TensorShape([cu_width, cu_height, 1]), tf.TensorShape([label_length]), tf.TensorShape([1]), tf.TensorShape([1]), tf.TensorShape([label_length])))
    data = data.shuffle(VALIDSET_READSIZE)
    data = data.repeat()
    data = data.batch(MINI_BATCH_SIZE)
    iterator = data.make_one_shot_iterator()

    images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch = iterator.get_next()

    return images_batch, label_batch, qp_batch, min_RDcost_batch, RDcost_batch
