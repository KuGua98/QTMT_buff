import os
import data_info as di

INDEX_LIST_TRAIN = 164
INDEX_LIST_VALID = 26

QP_LIST = [22, 27, 32, 37]

train_info_path = "E:/QTMT/CUInfo_train/"
valid_info_path = "E:/QTMT/CUInfo_valid/"
train_yuv_path = "E:/QTMT/YUV_HIF_train/"
valid_yuv_path = "E:/QTMT/YUV_HIF_valid/"

train_exe_path = "E:/QTMT/script/train/train.txt"
valid_exe_path = "E:/QTMT/script/valid/valid.txt"

####################    train      ##########################
train_file = open(train_exe_path,'a')

yuv_train = open("E:/QTMT/script/yuv_train_list.txt")
yuvframe_train = open("E:/QTMT/script/yuvframe_train_list.txt")
yuvheight_train = open("E:/QTMT/script/yuvheight_train_list.txt")
yuvwidth_train = open("E:/QTMT/script/yuvwidth_train_list.txt")

yuv_train_list = yuv_train.read().split(',')
yuvheight_train_list = yuvheight_train.read().split(',')
yuvwidth_train_list = yuvwidth_train.read().split(',')
yuvframe_train_list = yuvframe_train.read().split(',')

assert len(yuv_train_list)==len(yuvframe_train_list)
assert len(yuv_train_list)==len(yuvheight_train_list)
assert len(yuv_train_list)==len(yuvwidth_train_list)
for qp in QP_LIST:
    for i in range(len(yuv_train_list)):
        qp = str(qp)
        yuv_name = str(yuv_train_list[i])
        yuv_height = str(yuvheight_train_list[i])
        yuv_width = str(yuvwidth_train_list[i])
        yuv_frame = str(yuvframe_train_list[i])
        line_to_write = 'EncoderApp.exe -c encoder_intra_vtm.cfg -i '+train_yuv_path+yuv_name+' -wdt '+yuv_width+' -hgt '+yuv_height\
                        +' -fr 50 -f '+yuv_frame+' -q '+qp+'\n'
        train_file.write(line_to_write)
train_file.write('pause')

train_file.close()
yuv_train.close()
yuvframe_train.close()
yuvheight_train.close()
yuvwidth_train.close()

