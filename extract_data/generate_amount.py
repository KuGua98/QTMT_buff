import h5py
import numpy as np

PATH = "D:/QTMT/AMOUNT_INFO/"

file_train = h5py.File(PATH+'amount__train.h5','a')
file_valid = h5py.File(PATH+'amount__valid.h5','a')

CU_WIDTH_LIST = [64,  32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8,  8,  4, 4,   4]

################ train

# amount_all = np.zeros([len(CU_HEIGHT_LIST)])
# seq_amount = 0
# for i in range(2):
#
#     file_name = 'amount__train'+str(i+1)+'.h5'
#     file = h5py.File(PATH+file_name, 'r')
#
#     for j in range(len(CU_HEIGHT_LIST)):
#         cu_width = CU_WIDTH_LIST[j]
#         cu_height = CU_HEIGHT_LIST[j]
#         cu_name = str(cu_width) + 'x' + str(cu_height)
#
#         train_buff = file['record_amount/'+cu_name]
#         seq_amount_buff = len(train_buff)-1
#         if j == 0:
#             index = seq_amount
#             seq_amount = seq_amount + seq_amount_buff
#
#         if i == 0:
#             train_total = file_train.create_dataset(cu_name, (seq_amount_buff, 1), maxshape=(None, 1), dtype='int')
#         else:
#             train_total = file_train[cu_name]
#             train_total.resize([seq_amount, 1])
#
#         train_total[index:seq_amount] = train_buff[0:seq_amount_buff]
#         amount_all[j] = amount_all[j] + train_buff[seq_amount_buff]
#
#         if i == 1:
#             train_total.resize([seq_amount+1, 1])
#             train_total[seq_amount] = amount_all[j]
#
# file_train.close()



################ valid

amount_all = np.zeros([len(CU_HEIGHT_LIST)])
seq_amount = 0
for i in range(2):

    file_name = 'amount__valid'+str(i+1)+'.h5'
    file = h5py.File(PATH+file_name, 'r')

    for j in range(len(CU_HEIGHT_LIST)):
        cu_width = CU_WIDTH_LIST[j]
        cu_height = CU_HEIGHT_LIST[j]
        cu_name = str(cu_width) + 'x' + str(cu_height)

        train_buff = file['record_amount/'+cu_name]
        seq_amount_buff = len(train_buff)-1
        if j == 0:
            index = seq_amount
            seq_amount = seq_amount + seq_amount_buff

        if i == 0:
            train_total = file_valid.create_dataset(cu_name, (seq_amount_buff, 1), maxshape=(None, 1), dtype='int')
        else:
            train_total = file_valid[cu_name]
            train_total.resize([seq_amount, 1])

        train_total[index:seq_amount] = train_buff[0:seq_amount_buff]
        amount_all[j] = amount_all[j] + train_buff[seq_amount_buff]

        if i == 1:
            train_total.resize([seq_amount+1, 1])
            train_total[seq_amount] = amount_all[j]

file_valid.close()