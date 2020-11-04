import sys
sys.path.append("../..")
import h5py
import warnings
warnings.filterwarnings("ignore")

CU_WIDTH_LIST = [64,  32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8,  8,  4, 4,   4]

LABEL_LENGTH_LIST = [2, 6, 6, 6, 6, 6, 6, 6, 6, 6]
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]

TRAIN_SEQS_AMOUNT = 40
VALID_SEQS_AMOUNT = 26
# TRAIN_SEQS_AMOUNT = 164
# VALID_SEQS_AMOUNT = 26

AMOUNT_INFO_PATH = "D:/QTMT/AMOUNT_INFO/"

def get_sample_details(index):
    CU_WIDTH = CU_WIDTH_LIST[index]
    CU_HEIGHT = CU_HEIGHT_LIST[index]
    IMAGES_LENGTH = IMAGES_LENGTH_LIST[index]
    # SAMPLE_LENGTH = SAMPLE_LENGTH_LIST[index]
    LABEL_LENGTH = LABEL_LENGTH_LIST[index]

    return  CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH


def get_train_data_size(cu_name):
    # h5f_train = h5py.File(TRAIN_AMOUNT_INFO_PATH+'amount_train.h5', 'r')
    h5f_train = h5py.File(AMOUNT_INFO_PATH+'amount__train.h5', 'r')
    seq_amount_list = h5f_train[cu_name]
    total_amount = 0

    for i in range(TRAIN_SEQS_AMOUNT):
        total_amount = seq_amount_list[i] + total_amount

    return seq_amount_list, total_amount


def get_valid_data_size(cu_name):
    h5f_valid = h5py.File(AMOUNT_INFO_PATH+'amount__valid.h5', 'r')
    seq_amount_list = h5f_valid[cu_name]
    total_amount = 0

    for i in range(VALID_SEQS_AMOUNT):
        total_amount = seq_amount_list[i] + total_amount

    return seq_amount_list, total_amount