import sys
sys.path.append("../..")
import h5py
import numpy as np


path = "D:/QTMT/CU_SAMPLE_TRAIN/"

# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
CU_WIDTH_LIST = [64,  32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8,  8,  4, 4,   4]

LABEL_LENGTH_LIST = [2, 6, 6, 6, 6, 6, 6, 6, 6, 6]
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]

fid_out = open("prob_file.txt",'w')

# 遍历每个size的CU
for i in range(10):

    if i == 0:   # 64x64
        mode_amount = 2
    else:
        mode_amount = 6

    # mode: 0~5
    amount_list = np.zeros([mode_amount])
    prob_list = np.zeros([mode_amount])

    cu_width = CU_WIDTH_LIST[i]
    cu_height = CU_HEIGHT_LIST[i]
    sample_length = SAMPLE_LENGTH_LIST[i]

    cu_name = str(cu_width)+'x'+str(cu_height)
    file_name = cu_name+'_train.h5'
    print("for "+cu_name)

    file = h5py.File(path+file_name,'r')
    data_set = file[cu_name]

    total_number = len(data_set)
    # 遍历当前sizeCU中的所有sample
    for j in range(total_number):
        mode = data_set[j][sample_length-1].astype(int)
        amount_list[mode] = amount_list[mode] + 1

    for q in range(mode_amount):
        prob_list[q] = amount_list[q] / total_number
        fid_out.write(str(prob_list[q]) + '  ')
        print("mode"+ str(q) +" prob is :"+str(prob_list[q]))
    fid_out.write('\n')

    file.close()

fid_out.close

