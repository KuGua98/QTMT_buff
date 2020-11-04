import os
import data_info as di

INDEX_LIST_TRAIN = list(range(0,164))
INDEX_LIST_VALID = list(range(164,190))
# INDEX_LIST_TEST = list(range(190,2))

QP_LIST = [22, 27, 32, 37]

train_info_path = "E:/QTMT/CUInfo_train/"
valid_info_path = "E:/QTMT/CUInfo_valid/"
train_yuv_path = "E:/QTMT/YUV_HIF_train/"
valid_yuv_path = "E:/QTMT/YUV_HIF_valid/"

train_exe_path = "E:/QTMT/script/train/train.txt"
valid_exe_path = "E:/QTMT/script/valid/valid.txt"


###################    valid      ##########################
valid_file = open(valid_exe_path,'a')

yuv_valid = open("E:/QTMT/script/yuv_valid_list.txt")
yuvframe_valid = open("E:/QTMT/script/yuvframe_valid_list.txt")
yuvheight_valid = open("E:/QTMT/script/yuvheight_valid_list.txt")
yuvwidth_valid = open("E:/QTMT/script/yuvwidth_valid_list.txt")

yuv_valid_list = yuv_valid.read().split(',')
yuvheight_valid_list = yuvheight_valid.read().split(',')
yuvwidth_valid_list = yuvwidth_valid.read().split(',')
yuvframe_valid_list = yuvframe_valid.read().split(',')

assert len(yuv_valid_list)==len(yuvframe_valid_list)
assert len(yuv_valid_list)==len(yuvheight_valid_list)
assert len(yuv_valid_list)==len(yuvwidth_valid_list)
for qp in QP_LIST:
    for i in range(len(yuv_valid_list)):
        qp = str(qp)
        yuv_name = str(yuv_valid_list[i])
        yuv_height = str(yuvheight_valid_list[i])
        yuv_width = str(yuvwidth_valid_list[i])
        yuv_frame = str(yuvframe_valid_list[i])
        line_to_write = 'EncoderApp.exe -c encoder_intra_vtm.cfg -i '+valid_yuv_path+yuv_name+' -wdt '+yuv_width+' -hgt '+yuv_height\
                        +' -fr 50 -f '+yuv_frame+' -q '+qp+'\n'
        valid_file.write(line_to_write)
valid_file.write('pause')

valid_file.close()
yuv_valid.close()
yuvframe_valid.close()
yuvheight_valid.close()
yuvwidth_valid.close()