import sys
sys.path.append("../..")
import h5py
from extract_data import get_details as gd
import data_info as di


TRAINSET_READSIZE = 100 # shuffle size
VALIDSET_READSIZE = 100

INDEX_LIST_TRAIN1 = 0
INDEX_LIST_TRAIN2 = 40
INDEX_LIST_VALID1 = 164
INDEX_LIST_VALID2 = 190
# INDEX_LIST_TRAIN = list(range(0,164))
# INDEX_LIST_VALID = list(range(164,190))

YUV_NAME_TRAIN_LIST_FULL = di.YUV_NAME_LIST_FULL[INDEX_LIST_TRAIN1:INDEX_LIST_TRAIN2]
YUV_WIDTH_TRAIN_LIST_FULL = di.YUV_WIDTH_LIST_FULL[INDEX_LIST_TRAIN1:INDEX_LIST_TRAIN2]
YUV_HEIGHT_TRAIN_LIST_FULL = di.YUV_HEIGHT_LIST_FULL[INDEX_LIST_TRAIN1:INDEX_LIST_TRAIN2]
YUV_NAME_VALID_LIST_FULL = di.YUV_NAME_LIST_FULL[INDEX_LIST_VALID1:INDEX_LIST_VALID2]
YUV_WIDTH_VALID_LIST_FULL = di.YUV_WIDTH_LIST_FULL[INDEX_LIST_VALID1:INDEX_LIST_VALID2]
YUV_HEIGHT_VALID_LIST_FULL = di.YUV_HEIGHT_LIST_FULL[INDEX_LIST_VALID1:INDEX_LIST_VALID2]
# YUV_NAME_TRAIN_LIST_FULL = di.YUV_NAME_LIST_FULL[:INDEX_LIST_TRAIN]
# YUV_WIDTH_TRAIN_LIST_FULL = di.YUV_WIDTH_LIST_FULL[:INDEX_LIST_TRAIN]
# YUV_HEIGHT_TRAIN_LIST_FULL = di.YUV_HEIGHT_LIST_FULL[:INDEX_LIST_TRAIN]
#
# YUV_NAME_VALID_LIST_FULL = di.YUV_NAME_LIST_FULL[INDEX_LIST_TRAIN:INDEX_LIST_VALID]
# YUV_WIDTH_VALID_LIST_FULL = di.YUV_WIDTH_LIST_FULL[INDEX_LIST_TRAIN:INDEX_LIST_VALID]
# YUV_HEIGHT_VALID_LIST_FULL = di.YUV_HEIGHT_LIST_FULL[INDEX_LIST_TRAIN:INDEX_LIST_VALID]

TRAIN_SAMPLE_PATH = "D:/QTMT/TRAIN_SAMPLE/"
VALID_SAMPLE_PATH = "D:/QTMT/VALID_SAMPLE/"

TRAIN_SAMPLE_OUT = "D:/QTMT/CU_SAMPLE_TRAIN/"
VALID_SAMPLE_OUT = "D:/QTMT/CU_SAMPLE_VALID/"


def generate_train_h5(CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH):

    ######################     h5文件读取为dataset       ###########################
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    CU_NAME = str(CU_WIDTH) + 'x' + str(CU_HEIGHT)
    size_dataset_seq, size_dataset_all = gd.get_train_data_size(CU_NAME)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_train"
    cu_file = h5py.File(TRAIN_SAMPLE_OUT+CU_NAME+ MODE_NAME + '.h5', 'a')
    cu_dataset = cu_file.create_dataset(CU_NAME, (size_dataset_all, IMAGES_LENGTH+9), maxshape=(None, IMAGES_LENGTH+9), dtype='float32')
    index = 0

    for i in range(len(YUV_NAME_TRAIN_LIST_FULL)):
        yuv_name = YUV_NAME_TRAIN_LIST_FULL[i]
        h5_name =  'Samples_' + yuv_name + MODE_NAME + '.h5'
        h5_size = int(size_dataset_seq[i])

        h5_file = h5py.File(TRAIN_SAMPLE_PATH+h5_name, 'r')
        data_buff = h5_file[CU_NAME]

        cu_dataset[index:(index+h5_size)] = data_buff[0:h5_size]
        index = index + h5_size

        h5_file.close()

    cu_file.close()


def generate_valid_h5(CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH):

    ######################     h5文件读取为dataset       ###########################
    # buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
    CU_NAME = str(CU_WIDTH) + 'x' + str(CU_HEIGHT)
    size_dataset_seq, size_dataset_all = gd.get_valid_data_size(CU_NAME)  # 返回了所有序列该sizeCU的总数量，之后可能使用到
    MODE_NAME = "_valid"
    cu_file = h5py.File(VALID_SAMPLE_OUT+CU_NAME + MODE_NAME + '.h5', 'a')
    cu_dataset = cu_file.create_dataset(CU_NAME, (size_dataset_all, IMAGES_LENGTH + 9),
                                        maxshape=(None, IMAGES_LENGTH + 9), dtype='float32')
    index = 0

    for i in range(len(YUV_NAME_VALID_LIST_FULL)):
        yuv_name = YUV_NAME_VALID_LIST_FULL[i]
        h5_name = 'Samples_' + yuv_name + MODE_NAME + '.h5'
        h5_size = int(size_dataset_seq[i])

        h5_file = h5py.File(VALID_SAMPLE_PATH + h5_name, 'r')
        data_buff = h5_file[CU_NAME]

        cu_dataset[index:(index+h5_size)] = data_buff[0:h5_size]
        index = index + h5_size

        h5_file.close()

    cu_file.close()

if __name__ == '__main__':
    # IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
    for index in range(10):
        CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH = gd.get_sample_details(index)
        generate_train_h5(CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH)
        # generate_valid_h5(CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH)