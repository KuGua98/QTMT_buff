import sys
sys.path.append("..")
import tensorflow as tf
import net_cu as net
import h5py
import input_data_rect as input_data
from extract_data import get_details as gd
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

LEARNING_RATE_INIT = 1e-4
DECAY_RATE = 0.99
DECAY_STEP = 2000

MINI_BATCH_SIZE = 32
NUM_CHANNELS = 1

ITER_TIMES = 500000
ITER_TIMES_PER_EVALUATE = 100   # 100次迭代评估一次准确率
ITER_TIMES_PER_SAVE_ACCURACY = 1000  # 1000次迭代记录一次准确率
ITER_TIMES_PER_SAVE_MODEL = 10000 # 迭代10000次存储一次模型

TIMES_PER_COUNT_ACCURACY = 10   # 20*32=640 准确率计算分母为640

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


######################     getfile: 32x16       ###########################
# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
index = 3
CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, _ = gd.get_sample_details(index)
# For 32x16 : 0      2  3  4  5
LABEL_LENGTH = 5

images_batch_train, label_batch_train, qp_batch_train \
                                    = input_data.get_train_dataset(CU_WIDTH, CU_HEIGHT, LABEL_LENGTH, IMAGES_LENGTH)
images_batch_valid, label_batch_valid, qp_batch_valid \
                                    = input_data.get_valid_dataset(CU_WIDTH, CU_HEIGHT, LABEL_LENGTH)


######################     net: 32x16       ###########################

x = tf.placeholder("float", [None, CU_WIDTH, CU_HEIGHT, NUM_CHANNELS])
y = tf.placeholder("float", [None, LABEL_LENGTH])
qp = tf.placeholder("float", [None, 1])
global_step = tf.placeholder("float")

y_probabilty, y_predict, y_one_hot, total_loss_32x16, accuracy_32x16, learning_rate_current, train_step, opt_vars_all, \
    opt_vars_res4 = net.net_16x16_32x16(x, y, qp, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)


######################     feed dict       ###########################

# saver = tf.train.import_meta_graph()
# saver.restore(sess,数据路径)

config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

accuracy_file = open("Accuracy/accuracy_list_" + str(CU_WIDTH) +'x'+ str(CU_HEIGHT) + '.txt', 'a')

for i in range(ITER_TIMES):
    step = i + 1
    feed_step = step

    with tf.device('/gpu:0'):
        images_input, lable_input, qp_input =  sess.run([images_batch_train, label_batch_train, qp_batch_train])

    with tf.device('/gpu:0'):
        _, learning_rate, loss, accuracy = sess.run([train_step, learning_rate_current, total_loss_32x16, accuracy_32x16],feed_dict={x: images_input, y: lable_input, qp: qp_input,global_step: feed_step})

    if step % ITER_TIMES_PER_EVALUATE == 0:
        accuracy_32x16_list = []
        loss_32x16_list = []
        j = 0
        for j in range(TIMES_PER_COUNT_ACCURACY):

            images_input, lable_input, qp_input = sess.run([images_batch_valid, label_batch_valid, qp_batch_valid])
            # 验证时不更新网络参数
            loss, accuracy = sess.run([total_loss_32x16, accuracy_32x16],feed_dict={x: images_input, y: lable_input, qp: qp_input, global_step: feed_step})
            loss_32x16_list.append(loss)
            accuracy_32x16_list.append(accuracy)

        loss = sum(loss_32x16_list[0:TIMES_PER_COUNT_ACCURACY]) / TIMES_PER_COUNT_ACCURACY
        accuracy = sum(accuracy_32x16_list[0:TIMES_PER_COUNT_ACCURACY]) / TIMES_PER_COUNT_ACCURACY
        print("The " + str(step) + " times : loss is " + str(loss) + ",  accuracy is " + str(accuracy) + ", learning_rate is "+str(learning_rate))

    if step % ITER_TIMES_PER_SAVE_ACCURACY == 0:
        index = (step / ITER_TIMES_PER_SAVE_ACCURACY) - 1
        accuracy_file.write(str(accuracy)+'  '+str(loss)+'\n')
#
#     if step % ITER_TIMES_PER_SAVE_MODEL == 0:
#         saver.save(sess, 'Models_'+str(CU_WIDTH) +'x'+ str(CU_HEIGHT)+'/ing/model_'+str(CU_WIDTH) +'x'+ str(CU_HEIGHT)+'_%d.dat'%step)
#
# saver.save(sess, 'Models/models_'+str(CU_WIDTH) +'x'+ str(CU_HEIGHT)+'/final/model_'+str(CU_WIDTH) +'x'+ str(CU_HEIGHT)+'.dat')

accuracy_file.close()