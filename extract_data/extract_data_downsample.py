import sys
sys.path.append("..")
import os
import numpy as np
import random
import h5py
import data_info as di
import time


# ----------------------------------------------------------------------------------------------------------------------
# To configure: two variables
YUV_TRAIN_PATH_ORI = 'D:/QTMT/YUV_HIF_train/'  # path storing YUV files
INFO_TRAIN_PATH = 'D:/QTMT/script/train/dxc/1 f/' # path storing Info_XX.dat files for All-Intra configuration

YUV_VALID_PATH_ORI = 'D:/QTMT/YUV_HIF_valid/'  # path storing YUV files
INFO_VALID_PATH = 'D:/QTMT/SAMPLES_VALID/Vaild_All/' # path storing Info_XX.dat files for All-Intra configuration
# ----------------------------------------------------------------------------------------------------------------------

INDEX_LIST_TRAIN = 164
INDEX_LIST_VALID = 190
# INDEX_LIST_TRAIN = list(range(0,164))
# INDEX_LIST_VALID = list(range(164,190))

YUV_NAME_TRAIN_LIST_FULL = di.YUV_NAME_LIST_FULL[:INDEX_LIST_TRAIN]
YUV_WIDTH_TRAIN_LIST_FULL = di.YUV_WIDTH_LIST_FULL[:INDEX_LIST_TRAIN]
YUV_HEIGHT_TRAIN_LIST_FULL = di.YUV_HEIGHT_LIST_FULL[:INDEX_LIST_TRAIN]

YUV_NAME_VALID_LIST_FULL = di.YUV_NAME_LIST_FULL[INDEX_LIST_TRAIN:INDEX_LIST_VALID]
YUV_WIDTH_VALID_LIST_FULL = di.YUV_WIDTH_LIST_FULL[INDEX_LIST_TRAIN:INDEX_LIST_VALID]
YUV_HEIGHT_VALID_LIST_FULL = di.YUV_HEIGHT_LIST_FULL[INDEX_LIST_TRAIN:INDEX_LIST_VALID]

QP_LIST = [22, 27, 32, 37]
# QP_LIST = [22]

NAME_LIST = ['64x64','32x32','16x16','32x16','8x8','32x8','16x8','8x4','32x4','16x4']
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
# IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]
# RD_COST_LENGTH = 7
# LABEL_LENGTH = 1
# QP_LENGTH = 1

def get_file_list(yuv_path_ori, info_path, yuv_name_list, qp_list):
    yuv_file_list = []
    info_file_list = []

    for i_qp in range(len(qp_list)):
        info_file_list.append([])
        for i_seq in range(len(yuv_name_list)):
            info_file_list[i_qp].append(info_path+yuv_name_list[i_seq]+'_'+str(qp_list[i_qp]))

    for i_seq in range(len(yuv_name_list)):
        yuv_file_list.append(yuv_path_ori + yuv_name_list[i_seq])

    return yuv_file_list, info_file_list


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

def read_YUV420_frame(fid, width, height, i_frame):
    #  注意YUV420的数据存储格式
    #  read为连续读取
    d00 = height // 2
    d01 = width // 2

    # 注意每8帧编码一帧
    fid.seek((width * height + d01 * d00 * 2) * 8 * i_frame)
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8),[height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    return FrameYUV(Y, U, V)

def read_info_frame(fid, frame_index, cu_index, qp):
    data_len = 60   # 60个字节
    info = []
    icount = 0

    # 定位每一帧开始读的位置
    fid_start = cu_index*data_len
    fid.seek(fid_start)

    info_buf1 = np.frombuffer(fid.read(12), dtype = np.uint16).astype(np.float)
    info_buf2 = np.frombuffer(fid.read(48), dtype = np.float)
    info_buf = np.append(info_buf1, info_buf2)

    if len(info_buf1) != 0:
        while info_buf[0] == frame_index:
            # info_buf.append(qp)
            info_buf = np.append(info_buf, qp)
            info = np.append(info, info_buf)
            icount = icount + 1

            info_buf1 = np.frombuffer(fid.read(12), dtype=np.uint16).astype(np.float)
            info_buf2 = np.frombuffer(fid.read(48), dtype=np.float)
            info_buf = np.append(info_buf1, info_buf2)

            if len(info_buf1) == 0:
                break
    else:
        print("Have not coded complete")

    return info, icount


class Sample_Buff(object):
    def __init__(self, patch_Y, rd_cost, qp, partition_label):
        self._image_length = patch_Y.shape[0]
        self._patch_Y = patch_Y
        self._partition_label = partition_label
        self._qp = qp
        self._rd_cost = rd_cost

    @property
    def pacth_Y(self):
        return self._patch_Y

    @property
    def partition_label(self):
        return self._partition_label

    @property
    def qp(self):
        return self._qp

    @property
    def rd_cost(self):
        return self._rd_cost


def write_data(frame_Y, cu_info_list, h5f, amount_all_list, amount_record_list, MODE):
    cu_info_amount = cu_info_list.shape[0]
    amount_all_buff_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    amount_record_buff_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if MODE == 1:
        # 将所有CU数据量降到百万级
        strides_list = [1, 3, 5, 3, 5, 3, 10, 2, 1, 1]
    elif MODE == 2:
        # 验证集无需下采样
        strides_list = [1, 1, 1, 1 , 1 , 1 , 1, 1, 1, 1]

    for i in range(cu_info_amount):
        # 需要对数据做出筛选，丢弃不符合条件的数据
        choosed = 1
        notall_zero = 0
        for j in range(6,12):
            if cu_info_list[i][j] >= 999999999.9:
                # 丢弃过大的数据
                choosed = 0
            if cu_info_list[i][j] > 0:
                notall_zero = 1
        if notall_zero == 0:
            # 丢弃全为0的数据
            choosed = 0
        # 只选择亮度信息
        if cu_info_list[i][1] == 1:
            choosed = 0
        if cu_info_list[i][4] >= 128 or cu_info_list[i][5] >= 128:
            choosed = 0
        if choosed == 0:
            continue

        # 对符合条件的数据进行存储
        position_x = cu_info_list[i][2]
        position_y = cu_info_list[i][3]
        width = cu_info_list[i][4]
        height = cu_info_list[i][5]
        qp = cu_info_list[i][12]
        rdcost = []

        # 如果cu超出图片128整数倍的边界则跳过
        x_bound = frame_Y.shape[1] // 128 * 128
        y_bound = frame_Y.shape[0] // 128 * 128
        if width+position_x > x_bound or height+position_y > y_bound:
            continue

        for j in range(6, 12):
            rdcost.append(cu_info_list[i][j])
        patch_Y_buf = frame_Y[position_y.astype(int):(position_y+height).astype(int), position_x.astype(int):(position_x+width).astype(int)]

        # 对任何高大于宽的CU，需要先转置内容和分区模式
        if height > width:
            patch_Y_buf = np.transpose(patch_Y_buf)
            width, height = height, width
            rdcost[2], rdcost[3], rdcost[4], rdcost[5] = rdcost[3], rdcost[2], rdcost[5], rdcost[4]

        patch_Y = patch_Y_buf.reshape(1,(width*height).astype(int))
        rdcost = np.array(rdcost).reshape(1,6)

        # 记录rdcost的最小值和label
        rdcost_min = 999999999.9
        partition_model = -1
        for j in range(6):
            if rdcost[0][j] < rdcost_min and rdcost[0][j] != 0:
                rdcost_min = rdcost[0][j]
                partition_model = j
        assert partition_model != -1

        # 按CU size存储   buf_sample = patch_Y + rdcost + qp + rdcost_min + partition_model
        buf = np.append(patch_Y, rdcost)
        buf_sample = np.append(buf, [qp, rdcost_min, float(partition_model)]).astype(np.float32)

        cu_name = str(width.astype(int))+'x'+str(height.astype(int))

        #  DownSampling
        for i in range(len(NAME_LIST)):
            if cu_name == NAME_LIST[i]:
                length = SAMPLE_LENGTH_LIST[i]

                # 之后用来下采样
                index_buff = amount_all_list[i] + amount_all_buff_list[i]
                amount_all_buff_list[i] = amount_all_buff_list[i] + 1

                if (index_buff % strides_list[i]) != 0:
                    break

                index_real = amount_record_list[i] + amount_record_buff_list[i]
                amount_record_buff_list[i] = amount_record_buff_list[i] + 1

                data = h5f[cu_name]
                data[index_real] = buf_sample
                amount = len(data)

                if data[amount-1][length-3] != 0:
                    data.resize([amount+100, length])
                    h5f.flush()
                break

    return amount_all_buff_list, amount_record_buff_list


#
# def shuffle_samples():
#     cu_list = ['128x128','64x64','32x32','16x16','32x16','8x8','32x8',
#                '16x8','8x4','32x4','16x4']
#     #  数据存储格式为float64，一个占8个字节
#     length_list = [16393, 4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
#     # length_list = [16384, 4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]
#
#     for i in range(len(cu_list)):
#         file_name = cu_list[i]
#
#         # 获取sample数目
#         sample_length = length_list[i]*8
#         file_bytes = get_file_size(file_name)
#         assert file_bytes % sample_length == 0
#         num_samples = file_bytes // sample_length
#
#         # 文件改名
#         file_renamed = file_name+'.dat'
#         os.rename(file_name, file_renamed)
#
#         # 打乱
#         index_list = random.sample(range(num_samples),num_samples)
#         fid_in = open(file_renamed, 'rb')
#         fid_out = open(file_renamed+'_shuffled', 'wb')
#         for i in range(num_samples):
#             fid_in.seek((index_list[i]*sample_length), 0)
#             info_buf = fid_in.read(sample_length)
#             fid_out.write(info_buf)
#             if (i+1)%100 == 0 :
#                 print('%s : %d / %d samples completed.' % (file_renamed, i + 1, num_samples))
#         fid_in.close()
#         fid_out.close()


def generate_data(yuv_path_ori, info_path, yuv_name_list_full, yuv_width_list_full, yuv_height_list_full, qp_list,
                  MODE):
    yuv_file_list, info_file_list = get_file_list(yuv_path_ori, info_path, yuv_name_list_full, qp_list)
    n_seq = len(yuv_file_list)
    n_qp = len(qp_list)
    if MODE == 1:
        MODE_NAME = '_train'
    elif MODE == 2:
        MODE_NAME = '_valid'

    # 对整个数据集，记录其CU数量
    amount_all_list_dataset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    amount_record_list_dataset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    h5f_amount = h5py.File("amount_" + MODE_NAME + ".h5", 'a')
    h5f_total_amount = h5f_amount.create_group("total_amount")
    h5f_record_amount = h5f_amount.create_group("record_amount")

    for i in range(len(NAME_LIST)):
        name = NAME_LIST[i]
        length = n_seq+1
        h5f_total_amount.create_dataset(name, (length, 1), maxshape=(None, 1), dtype='float32')
        h5f_record_amount.create_dataset(name, (length, 1), maxshape=(None, 1), dtype='float32')

    for i_seq in range(n_seq):
        # 对每个序列，记录其CU的数量
        amount_all_list_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        amount_record_list_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        Sample_Name = yuv_name_list_full[i_seq]
        h5f_Sample = h5py.File('Samples_' + str(Sample_Name) + MODE_NAME + '.h5', 'a')
        for i in range(len(NAME_LIST)):
            name = NAME_LIST[i]
            length = SAMPLE_LENGTH_LIST[i]
            h5f_Sample.create_dataset(name, (100, length), maxshape=(None, length), dtype='float32')

        width = yuv_width_list_full[i_seq]
        height = yuv_height_list_full[i_seq]
        n_frame = get_num_YUV420_frame(yuv_file_list[i_seq] + '.yuv', width, height)
        fid_yuv = open(yuv_file_list[i_seq] + '.yuv', 'rb')

        print(yuv_name_list_full[i_seq] + '.yuv' + ': Generate Samples Begin ')

        fid_info_list = []
        for i_qp in range(n_qp):
            fid_info_list.append(open(info_file_list[i_qp][i_seq] + '.dat', 'rb'))

        # 因为不知道每一帧CU的个数，所以需要定位每次读取的开始位置
        cu_index = np.zeros([n_qp]).astype(int)

        # m_temporalSubsampleRatio=8时，每8帧编码1帧
        encoded_frame = (n_frame + 8 - 1) / 8
        for i_frame in range(int(encoded_frame)):

            print('%d frame' % i_frame)
            frame_YUV = read_YUV420_frame(fid_yuv, width, height, i_frame)
            frame_Y = frame_YUV._Y
            # frame_U = frame_YUV._U
            # frame_V = frame_YUV._V

            for i_qp in range(n_qp):
                cu_info_buff, cu_amount = read_info_frame(fid_info_list[i_qp], i_frame, cu_index[i_qp], qp_list[i_qp])
                cu_index[i_qp] = cu_index[i_qp] + cu_amount
                cu_info = np.reshape(cu_info_buff, (cu_amount, 13))
                amount_all, amount_record = write_data(frame_Y, cu_info, h5f_Sample, amount_all_list_seq,
                                                       amount_record_list_seq, MODE)
                # 记录每个序列各个size的CU数量
                for j in range(len(amount_all)):
                    amount_all_list_seq[j] = amount_all_list_seq[j] + amount_all[j]
                    amount_record_list_seq[j] = amount_record_list_seq[j] + amount_record[j]

        # 输出每个序列的CU数量信息
        print(yuv_name_list_full[i_seq] + '.yuv' + ': Generate Samples End ')
        for q in range(len(NAME_LIST)):
            print('%s : %d samples.' % (NAME_LIST[q], amount_all_list_seq[q]))
            print('%s : %d samples.' % (NAME_LIST[q], amount_record_list_seq[q]))
            total_amount = h5f_total_amount[NAME_LIST[q]]
            record_amount = h5f_record_amount[NAME_LIST[q]]
            total_amount[i_seq] = amount_all_list_seq[q]
            record_amount[i_seq] = amount_record_list_seq[q]

        # 记录整个数据集的CU数量
        for j in range(len(amount_all_list_seq)):
            amount_all_list_dataset[j] = amount_all_list_dataset[j] + amount_all_list_seq[j]
            amount_record_list_dataset[j] = amount_record_list_dataset[j] + amount_record_list_seq[j]

        h5f_Sample.close()

    # 输出整个数据集的CU数量信息
    for q in range(len(NAME_LIST)):
        print('%s : %d samples.' % (NAME_LIST[q], amount_all_list_dataset[q]))
        print('%s : %d samples.' % (NAME_LIST[q], amount_record_list_dataset[q]))
        total_amount = h5f_total_amount[NAME_LIST[q]]
        record_amount = h5f_record_amount[NAME_LIST[q]]
        total_amount[n_seq] = amount_all_list_dataset[q]
        record_amount[n_seq] = amount_record_list_dataset[q]

    h5f_amount.close()


if __name__ == '__main__':

    time_start = time.time()
    # For train
    # generate_data(YUV_TRAIN_PATH_ORI, INFO_TRAIN_PATH, YUV_NAME_TRAIN_LIST_FULL, YUV_WIDTH_TRAIN_LIST_FULL, YUV_HEIGHT_TRAIN_LIST_FULL, QP_LIST, 1)

    # For valid
    generate_data(YUV_VALID_PATH_ORI, INFO_VALID_PATH, YUV_NAME_VALID_LIST_FULL, YUV_WIDTH_VALID_LIST_FULL, YUV_HEIGHT_VALID_LIST_FULL, QP_LIST, 2)

    time_end = time.time()

    print("time_cost: ", time_end-time_start , "s")
