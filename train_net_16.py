import sys
sys.path.append("..")
import tensorflow as tf
import net_cu as net
import input_data_16 as input_data
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
ITER_TIMES_PER_EVALUATE = 100   # 50次迭代评估一次准确率
ITER_TIME_PER_COUNT_ACCURACY = 20   # 20*32=640 准确率计算分母为640
ITER_TIMES_PER_SAVE = 1000
ITER_TIMES_PER_SAVE_MODEL = 10000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


######################     getfile: 16x16 or 32x16       ###########################
# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
index = 2
CU_WIDTH_1, CU_HEIGHT_1, IMAGES_LENGTH_1, LABEL_LENGTH_1 = gd.get_sample_details(index)

images_batch_train_1, label_batch_train_1, qp_batch_train_1 \
                                    = input_data.get_train_dataset1(CU_WIDTH_1, CU_HEIGHT_1, LABEL_LENGTH_1, IMAGES_LENGTH_1)
images_batch_valid_1, label_batch_valid_1, qp_batch_valid_1 \
                                    = input_data.get_valid_dataset1(CU_WIDTH_1, CU_HEIGHT_1, LABEL_LENGTH_1)

index = 3
CU_WIDTH_2, CU_HEIGHT_2, IMAGES_LENGTH_2, LABEL_LENGTH_2 = gd.get_sample_details(index)
images_batch_train_2, label_batch_train_2, qp_batch_train_2 \
                                    = input_data.get_train_dataset2(CU_WIDTH_2, CU_HEIGHT_2, LABEL_LENGTH_2, IMAGES_LENGTH_2)
images_batch_valid_2, label_batch_valid_2, qp_batch_valid_2 \
                                    = input_data.get_valid_dataset2(CU_WIDTH_2, CU_HEIGHT_2, LABEL_LENGTH_2)


######################     net: 16x16 or 32x16       ###########################
global_step = tf.placeholder("float")
# y_probabilty, y_predict, y_one_hot, total_loss_16x16_train, accuracy_16x16_train, learning_rate_current, train_step, opt_vars_all, \
#     opt_vars_res4 = net.net_16x16_32x16(images_batch_train_1, label_batch_train_1, qp_batch_train_1, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)
_, _, _, total_loss_16x16_train, accuracy_16x16_train, learning_rate_current_16x16_train, train_step_16x16, _, _ = \
    net.net_16x16_32x16(images_batch_train_1, label_batch_train_1, qp_batch_train_1, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)
_, _, _, total_loss_16x16_valid, accuracy_16x16_valid, learning_rate_current_16x16_valid, _, opt_vars_all_16x16, _ = \
    net.net_16x16_32x16(images_batch_valid_1, label_batch_valid_1, qp_batch_valid_1, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)

_, _, _, total_loss_32x16_train, accuracy_32x16_train, learning_rate_current_32x16_train, train_step_32x16, _, _ = \
    net.net_16x16_32x16(images_batch_train_2, label_batch_train_2, qp_batch_train_2, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)
_, _, _, total_loss_32x16_valid, accuracy_32x16_valid, learning_rate_current_32x16_valid, _, opt_vars_all_32x16, opt_vars_res4 = \
    net.net_16x16_32x16(images_batch_valid_2, label_batch_valid_2, qp_batch_valid_2, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)


######################     feed dict       ###########################

saver_16x16 = tf.train.Saver(opt_vars_all_16x16, write_version=tf.train.SaverDef.V2)
saver_32x16 = tf.train.Saver(opt_vars_all_32x16, write_version=tf.train.SaverDef.V2)
saver_res4 = tf.train.Saver(opt_vars_res4, write_version=tf.train.SaverDef.V2)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
for i in range(ITER_TIMES):

    step = i + 1
    feed_step = step

    time_end_input = time.time()

    # with tf.device('/gpu:0'):
    _, learning_rate, loss, accuracy = sess.run([train_step_16x16, learning_rate_current_16x16_train, total_loss_16x16_train, accuracy_16x16_train],  feed_dict={global_step: feed_step})
    _, _, _, _ = sess.run([train_step_32x16, learning_rate_current_32x16_train, total_loss_32x16_train, accuracy_32x16_train],  feed_dict={global_step: feed_step})

    time_end_calcu = time.time()
    time2 = time_end_calcu - time_end_input

    print(time_end_input, time_end_calcu)
    print("cal time: %d - %d = %d"%(time_end_input, time_end_calcu, time2))
    print(learning_rate, loss, accuracy)

    if step % ITER_TIMES_PER_EVALUATE == 0:
        accuracy_16x16_list = []
        loss_16x16_list = []
        accuracy_32x16_list = []
        loss_32x16_list = []
        j = 0
        for j in range(ITER_TIME_PER_COUNT_ACCURACY):
            # 验证时不更新网络参数
            learning_rate_16x16, loss_16x16, accuracy_16x16 = sess.run([learning_rate_current_16x16_train, total_loss_16x16_train, accuracy_16x16_train], feed_dict={global_step: feed_step})
            learning_rate_32x16, loss_32x16, accuracy_32x16 = sess.run([learning_rate_current_32x16_train, total_loss_32x16_train, accuracy_32x16_train], feed_dict={global_step: feed_step})

            loss_16x16_list.append(loss_16x16)
            accuracy_16x16_list.append(accuracy_16x16)

            loss_32x16_list.append(loss_32x16)
            accuracy_32x16_list.append(accuracy_32x16)

        loss_16x16 = sum(loss_16x16_list[0:ITER_TIME_PER_COUNT_ACCURACY]) / ITER_TIME_PER_COUNT_ACCURACY
        accuracy_16x16 = sum(accuracy_16x16_list[0:ITER_TIME_PER_COUNT_ACCURACY]) / ITER_TIME_PER_COUNT_ACCURACY
        print("For 16x16 , The " + str(i) + " times : loss is " + str(loss_16x16) + ",  accuracy is " + str(accuracy_16x16) + ", learning_rate is "+str(learning_rate_16x16))

        loss_32x16 = sum(loss_32x16_list[0:ITER_TIME_PER_COUNT_ACCURACY]) / ITER_TIME_PER_COUNT_ACCURACY
        accuracy_32x16 = sum(accuracy_32x16_list[0:ITER_TIME_PER_COUNT_ACCURACY]) / ITER_TIME_PER_COUNT_ACCURACY
        print("For 32x16 , The " + str(i) + " times : loss is " + str(loss_32x16) + ",  accuracy is " + str(accuracy_32x16) + ", learning_rate is "+str(learning_rate_32x16))

        # weight = sess.graph.get_tensor_by_name('w_subnet_nc_64_1')
        # bia = sess.graph.get_tensor_by_name('b_subnet_nc_64_1')
        # w = sess.run(weight)
        # b = sess.run(bia)
        # print(y)

#     if step % ITER_TIMES_PER_SAVE == 0:
#         saver.save(sess, 'Models/ing/model_64x64_%d.dat'%step)
#         saver_res1.save(sess, 'Models/ing/model_res1_%d.dat' % step)
#         saver_res2.save(sess, 'Models/ing/model_res2_%d.dat' % step)
#
# saver.save(sess, 'Models/final/model_64x64.dat')
# saver_res1.save(sess, 'Models/final/model_res1.dat')
# saver_res2.save(sess, 'Models/final/model_res2.dat')