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

# 遍历每个size的CU
# for i in range(10):
i=1

if i == 0:   # 64x64
    mode_amount = 2
else:
    mode_amount = 6

# mode: 0~5
amount_list = 0

cu_width = CU_WIDTH_LIST[i]
cu_height = CU_HEIGHT_LIST[i]
image_length = IMAGES_LENGTH_LIST[i]
sample_length = SAMPLE_LENGTH_LIST[i]

cu_name = str(cu_width)+'x'+str(cu_height)

fid_out = open(cu_name+"valid_amount.txt",'w')
file_name = cu_name+'_train.h5'
print("for "+cu_name)

file = h5py.File(path+file_name,'r')
data_set = file[cu_name]

total_number = len(data_set)
# 遍历当前sizeCU中的所有sample
for j in range(total_number):
    choosed=1

    for q in range(6):
        rdcost = data_set[j][image_length+q].astype(int)
        if rdcost == 0:
            choosed=0
            break
    if choosed==1:
        amount_list += 1
    if j%1000 == 0 :
        print("%d samples have completed, %d samples is valid"%(j+1, amount_list))


print("have %d valid samples!"%amount_list)
fid_out.write(amount_list)

file.close()

fid_out.close

