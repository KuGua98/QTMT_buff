import numpy as np
import os
import sys
sys.path.append('C:/Users/1/Desktop/pzx/VVC/VVC_test/MSE-CNN_Training_AI/extract_data')
import data_info as di

YUV_INPUT = 'E:/QTMT/YUV_HIF/'   # 带裁切序列路径
YUV_OUTPUT = 'E:/QTMT/YUV_HIF_CUTTED/'  #  已裁切序列路径

YUV_NAME_LIST_FULL = di.YUV_NAME_LIST_FULL
YUV_WIDTH_LIST_FULL = di.YUV_WIDTH_LIST_FULL
YUV_HEIGHT_LIST_FULL = di.YUV_HEIGHT_LIST_FULL


def get_file_list(yuv_path_ori, yuv_name_list):
    yuv_file_list = []
    for i_seq in range(len(yuv_name_list)):
        yuv_file_list.append(yuv_path_ori + yuv_name_list[i_seq])
    return yuv_file_list


class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V


def get_file_size(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


def get_num_YUV420_frame(file, width, height):
    file_bytes = get_file_size(file)
    # 格式为YUV420， Y:U:V=4:2:0
    frame_bytes = width * height * 3 // 2
    assert(file_bytes % frame_bytes == 0)
    frame_number = file_bytes // frame_bytes
    return frame_number


def read_YUV420_frame(fid, width, height):
    #  注意YUV420的数据存储格式
    #  read为连续读取
    d00 = height // 2
    d01 = width // 2
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8),[height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    return FrameYUV(Y, U, V)


def write_yuv(frame_YUV, width, height, fid_yuv_out):
    frame_Y = frame_YUV._Y
    frame_U = frame_YUV._U
    frame_V = frame_YUV._V

    # 取整之后再乘以128
    h = height // 128 * 128
    w = width // 128 * 128
    d00 = h // 2
    d01 = w // 2

    # 注意坐标（x,y）和尺寸(w,h) 是相反的
    Y = frame_Y[0:h, 0:w]
    U = frame_U[0:d00,0:d01]
    V = frame_V[0:d00,0:d01]

    # frame = np.concatenate(np.concatenate(Y,U),V)

    fid_yuv_out.write(np.ascontiguousarray(Y))
    fid_yuv_out.write(np.ascontiguousarray(U))
    fid_yuv_out.write(np.ascontiguousarray(V))


if __name__ == '__main__':
    n_seq = len(YUV_NAME_LIST_FULL)

    for i_seq in range(n_seq):
        yuv_name = YUV_NAME_LIST_FULL[i_seq]
        yuv_file = YUV_INPUT + yuv_name + '.yuv'
        width =  YUV_WIDTH_LIST_FULL[i_seq]
        height = YUV_HEIGHT_LIST_FULL[i_seq]
        n_frame = get_num_YUV420_frame(yuv_file, width, height)
        fid_yuv = open(yuv_file, 'rb')
        fid_yuv_out = open(YUV_OUTPUT + yuv_name + '.yuv', 'wb+')

        for i_frame in range(n_frame):
            frame_YUV = read_YUV420_frame(fid_yuv, width, height)
            write_yuv(frame_YUV, width , height, fid_yuv_out)

        fid_yuv_out.close()