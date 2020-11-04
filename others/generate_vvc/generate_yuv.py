import os
import xlrd

# YuvPath = 'E:\QTMT\YUV_HIF_valid'
YuvPath = 'E:\QTMT\YUV_HIF_train'


########## 读取序列信息：名字 宽 高 帧数################
sequence_list = []
# 打开文件
data = xlrd.open_workbook('C:/Users/1/Desktop/pzx/VVC/VVC_test/MSE-CNN_Training_AI/generate_vvc/sequence_info.xlsx')
# 查看工作表
data.sheet_names()
# 通过文件名获得工作表,获取工作表1
table = data.sheet_by_name('Sheet2')

for i in range(table.nrows):
    sequence_list.append([])
    for j in range(table.ncols):
        sequence_list[i].append(table.cell(i,j).value)
#####################################################


##########读取带.yuv的文件名字############
# file = open("yuv_valid_list.txt",'w')
file = open("yuv_train_list.txt",'w')
for yuv in os.listdir(YuvPath):
    file.write(yuv)
    file.write(",")


Namefile = open("yuvname_train_list.txt", 'w')
Widthfile = open("yuvwidth_train_list.txt",'w')
Heightfile = open("yuvheight_train_list.txt",'w')
Framefile = open('yuvframe_train_list.txt','w')
# Namefile = open("yuvname_valid_list.txt", 'w')
# Widthfile = open("yuvwidth_valid_list.txt",'w')
# Heightfile = open("yuvheight_valid_list.txt",'w')
# Framefile = open('yuvframe_valid_list.txt','w')
icount = 0
for yuv in os.listdir(YuvPath):
    yuv_name = yuv.replace('.yuv','')
    for i in range(table.nrows):
        if yuv_name == sequence_list[i][0]:
            icount = icount + 1
            print(icount)

            yuv_width = sequence_list[i][1]
            yuv_height = sequence_list[i][2]
            yuv_frames = sequence_list[i][3]

            Framefile.write(str(int(yuv_frames)))
            Framefile.write(',')

            Widthfile.write(str(int(yuv_width)))
            Widthfile.write(',')

            Heightfile.write(str(int(yuv_height)))
            Heightfile.write(',')


            Namefile.write('\''+yuv_name+'\',\n')
            # Namefile.write(',\n')

file.close()
Namefile.close()
Widthfile.close()
Heightfile.close()
Framefile.close()
