import os
import numpy as np
import random
import h5py
import data_info as di
import time


INFO_TRAIN_PATH = 'E:/QTMT/CUInfo_train/'

data_len = 60  # 60个字节
info = []


PATH = INFO_TRAIN_PATH+'akiyo_cif_22.dat'

file = open(PATH, 'rb')
fid = open("info_test.txt",'wb')

size = os.path.getsize(PATH)
# size = get_file_size(file)

amount = size / 60

last_poc = 0
# 定位每一帧开始读的位置
for cu_index in range(int(amount)):

    info_buf1 = np.reshape(np.frombuffer(file.read(12), dtype=np.uint16), [1,6])
    info_buf2 = np.reshape(np.frombuffer(file.read(48), dtype=np.float), [1,6])

    now_poc = info_buf1[0][0]
    channel = info_buf1[0][1]
    position_x = info_buf1[0][2]
    position_y = info_buf1[0][3]
    width = info_buf1[0][4]
    height = info_buf1[0][5]
    rdcost = []

    for j in range(0, 6):
        rdcost.append(info_buf2[0][j])

    buf = np.append(now_poc, [channel, position_x, position_y, width, height]).astype(np.float32)
    info = np.append(buf, rdcost).astype(np.float32)


    fid.write(info)

    # if last_poc !=  now_poc:
    #     last_poc = now_poc
    # print(info_buf1[0])