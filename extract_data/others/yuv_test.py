import os
import numpy as np
import random
import h5py
import data_info as di
import time


YUV_TRAIN_PATH = 'E:/QTMT/YUV_HIF_train/'
PATH = YUV_TRAIN_PATH+'akiyo_cif.yuv'

width = 352
height = 288

fid = open(PATH, 'rb')

file_bytes = os.path.getsize(PATH)
# size = get_file_size(file)

frame_bytes = width * height * 3 // 2
assert(file_bytes % frame_bytes == 0)
frame_number = file_bytes // frame_bytes

d00 = height // 2
d01 = width // 2


class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V

# 注意每8帧编码一帧
for i_frame in range(frame_number):
    fid.seek((width * height + d01 * d00 * 2)* i_frame)
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    if i_frame == 0:
        Y_0 = Y
    if i_frame == 8:
        Y_1 = Y
        Y_diff = Y_1 - Y_0
