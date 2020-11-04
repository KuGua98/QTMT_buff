import tensorflow as tf
import resNet as res
import subNet as sub
import numpy as np
import math

DEFAULT_THR_LIST = [[0.75, 0.55, 0.55, 0.55, 0.6],
                    [0.65, 0.45, 0.45, 0.45, 0.5],
                    [0.5,  0.4,  0.35, 0.35, 0.4],
                    [0.45, 0.3,  0.25, 0.25, 0.25],
                    [0.3,  0.2,  0.2,  0.15, 0.15]]

adjust_scalar_64 = [-0.3, -0.3]      # α=0.5  要对应数量以array形式输入
adjust_scalar_else = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3,]
positive_scalar = 0.5    # β

ITER_TIMES = 500000

# 分类标签的数量
NUM_CLASSES_64X64 = 2
NUM_CLASSES_OTHERS = 6

# 不同大小的CU的不同模式所占的比例
# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
p_64x64 = [0.25, 0.75]
p_32x32 = [0.22, 0.21, 0.24, 0.19, 0.08, 0.06]
p_16x16 = [0.34, 0.06, 0.23, 0.24, 0.07, 0.06]
p_32x16 = [0.45, 0.00, 0.20, 0.22, 0.04, 0.09]
p_8x8 = []
p_32x8 = []
p_16x8 = []
p_8x4 = []
p_32x4 = []
p_16x4 = []


def net_64x64(x, y, qp, global_step, learning_rate_init, decay_rate, decay_step):
    # 归一化
    x = tf.cast(x, tf.float32)
    x = tf.scalar_mul(1.0 / 255.0, x)
    x_image = tf.reshape(x, [-1, 64, 64, 1])
    y_image = tf.reshape(y, [-1, 2])

    h_cov = sub.overlap_conv(x_image, 3, 3, 1, 16)
    h_condc = res.condc_lumin_64(h_cov)
    y_probabilty = sub.sub_net_64(h_condc, qp)  # 预测的概率值

    y_predict =  tf.argmax(y_probabilty, axis=1)  # 返回最大值索引
    y_one_hot = tf.one_hot(indices=y_predict, depth=2)  # 转换为one—hot vector

    loss_64_ce = -tf.reduce_sum(tf.multiply(np.power(p_64x64, adjust_scalar_64).astype(np.float32), tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_64x64, adjust_scalar_64))
    # loss_64_rd = tf.reduce_sum(-tf.multiply(y_image, tf.log()))
    # total_loss_64x64 = loss_64_ce + positive_scalar*loss_64_rd
    total_loss_64x64 = loss_64_ce

    accuracy_64x64 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
    learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_64x64)

    opt_vars_all = [v for v in tf.trainable_variables()]
    opt_vars_res1 = tf.trainable_variables(scope='res_unit1')
    opt_vars_res2 = tf.trainable_variables(scope='res_unit2')

    # accuracy_64x64_list = tf.stack(accuracy_64x64)
    # loss_64x64_list = tf.stack(total_loss_64x64)

    return y_probabilty, y_predict, y_one_hot, total_loss_64x64, accuracy_64x64, learning_rate_current, train_step, \
           opt_vars_all, opt_vars_res1, opt_vars_res2
    # return y_probabilty, y_predict, y_one_hot, total_loss_64x64, loss_64x64_list, \
    #        learning_rate_current, train_step, accuracy_64x64_list, opt_vars_all


def net_32x32(x, y, qp, global_step, learning_rate_init, decay_rate, decay_step):
    # 归一化
    x = tf.scalar_mul(1.0 / 255.0, x)
    x_image = tf.reshape(x, [-1, 32, 32, 1])
    y_image = tf.reshape(y, [-1, 6])

    h_cov = sub.overlap_conv(x_image, 3, 3, 1, 16)
    h_condc = res.condc_lumin_32(h_cov)
    y_probabilty = sub.sub_net_32(h_condc, qp)

    y_predict = tf.argmax(y_probabilty, axis=1)  # 返回最大值索引
    y_one_hot = tf.one_hot(indices=y_predict, depth=6)  # 转换为one—hot vector

    loss_32_ce = -tf.reduce_sum(tf.multiply(np.power(p_32x32, adjust_scalar_else).astype(np.float32),tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_32x32, adjust_scalar_else))

    total_loss_32x32 = loss_32_ce

    accuracy_32x32 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
    learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_32x32)

    opt_vars_all = [v for v in tf.trainable_variables()]
    opt_vars_res3 = tf.trainable_variables(scope='res_unit3')

    return y_probabilty, y_predict, y_one_hot, total_loss_32x32, accuracy_32x32, learning_rate_current, train_step, opt_vars_all, opt_vars_res3



def net_16x16_32x16(x, y, qp, global_step, learning_rate_init, decay_rate, decay_step):
    # 归一化
    CU_WIDTH = int(x.shape[1])
    CU_HEIGHT = int(x.shape[2])
    x = tf.scalar_mul(1.0 / 255.0, x)
    if CU_WIDTH==16:
        x_image = tf.reshape(x, [-1, 16, 16, 1])
        y_image = tf.reshape(y, [-1, 6])
    elif CU_WIDTH==32:
        x_image = tf.reshape(x, [-1, 32, 16, 1])
        y_image = tf.reshape(y, [-1, 6])

    h_cov = sub.overlap_conv(x_image, 3, 3, 1, 16)
    h_condc = res.condc_lumin_16(h_cov)
    y_probabilty = sub.sub_net_16(h_condc, CU_WIDTH, CU_HEIGHT, qp)
    y_predict = tf.argmax(y_probabilty, axis=1)  # 返回最大值索引

    if CU_WIDTH == 16:
        y_one_hot = tf.one_hot(indices=y_predict, depth=6)  # 转换为one—hot vector
        loss_16x16_ce = -tf.reduce_sum(tf.multiply(np.power(p_16x16, adjust_scalar_else).astype(np.float32),tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_16x16, adjust_scalar_else))
        total_loss_16x16 = loss_16x16_ce
        accuracy_16x16 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        with tf.variable_scope('adam', reuse=tf.AUTO_REUSE):
            train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_16x16)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res4 = tf.trainable_variables(scope='res_unit4')

        return y_probabilty, y_predict, y_one_hot, total_loss_16x16, accuracy_16x16, learning_rate_current, train_step, opt_vars_all, opt_vars_res4

    elif CU_WIDTH == 32:
        y_one_hot = tf.one_hot(indices=y_predict, depth=6)  # 转换为one—hot vector
        loss_32x16_ce = -tf.reduce_sum(tf.multiply(np.power(p_32x16, adjust_scalar_else).astype(np.float32), tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_32x16, adjust_scalar_else))
        total_loss_32x16 = loss_32x16_ce
        accuracy_32x16 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        with tf.variable_scope('adam', reuse=tf.AUTO_REUSE):
            train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_32x16)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res4 = tf.trainable_variables(scope='res_unit4')

        return y_probabilty, y_predict, y_one_hot, total_loss_32x16, accuracy_32x16, learning_rate_current, train_step, opt_vars_all, opt_vars_res4


def net_8x8_16x8_32x8(x, y, qp, global_step, learning_rate_init, decay_rate, decay_step):
    # 归一化
    CU_WIDTH = int(x.shape[1])
    CU_HEIGHT = int(x.shape[2])
    x = tf.scalar_mul(1.0 / 255.0, x)
    if CU_WIDTH==8:
        x_image = tf.reshape(x, [-1, 8, 8, 1])
        y_image = tf.reshape(y, [-1, 6])
    elif CU_WIDTH==16:
        x_image = tf.reshape(x, [-1, 16, 8, 1])
        y_image = tf.reshape(y, [-1, 6])
    elif CU_WIDTH == 32:
        x_image = tf.reshape(x, [-1, 32, 8, 1])
        y_image = tf.reshape(y, [-1, 6])

    h_cov = sub.overlap_conv(x_image, 3, 3, 1, 16)
    h_condc = res.condc_lumin_8(h_cov)
    y_probabilty = sub.sub_net_8x8_16x8_32x8(h_condc, qp)
    y_predict = tf.argmax(y_probabilty, axis=1)  # 返回最大值索引

    if CU_WIDTH == 8:
        y_one_hot = tf.one_hot(indices=y_predict, depth=3)  # 转换为one—hot vector
        loss_8x8_ce = -tf.reduce_sum(tf.multiply(np.power(p_8x8, adjust_scalar_else).astype(np.float32),tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_8x8, adjust_scalar_else))
        total_loss_8x8 = loss_8x8_ce
        accuracy_8x8 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_8x8)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res5 = tf.trainable_variables(scope='res_unit5')

        return y_probabilty, y_predict, y_one_hot, total_loss_8x8, accuracy_8x8, learning_rate_current, train_step, opt_vars_all, opt_vars_res5

    elif CU_WIDTH == 16:
        y_one_hot = tf.one_hot(indices=y_predict, depth=4)  # 转换为one—hot vector
        loss_16x8_ce = -tf.reduce_sum(tf.multiply(np.power(p_16x8, adjust_scalar_else).astype(np.float32), tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_16x8, adjust_scalar_else))
        total_loss_16x8 = loss_16x8_ce
        accuracy_16x8 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_16x8)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res6 = tf.trainable_variables(scope='res_unit5')

        return y_probabilty, y_predict, y_one_hot, total_loss_16x8, accuracy_16x8, learning_rate_current, train_step, opt_vars_all, opt_vars_res6

    elif CU_WIDTH == 32:
        y_one_hot = tf.one_hot(indices=y_predict, depth=4)  # 转换为one—hot vector
        loss_32x8_ce = -tf.reduce_sum(tf.multiply(np.power(p_32x8, adjust_scalar_else).astype(np.float32), tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_32x8, adjust_scalar_else))
        total_loss_32x8 = loss_32x8_ce
        accuracy_32x8 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_32x8)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res5 = tf.trainable_variables(scope='res_unit5')

        return y_probabilty, y_predict, y_one_hot, total_loss_32x8, accuracy_32x8, learning_rate_current, train_step, opt_vars_all, opt_vars_res5


def net_8x4_16x4_32x4(x, y, qp, global_step, learning_rate_init, decay_rate, decay_step):
    # 归一化
    CU_WIDTH = int(x.shape[1])
    CU_HEIGHT = int(x.shape[2])
    x = tf.scalar_mul(1.0 / 255.0, x)
    if CU_WIDTH==8:
        x_image = tf.reshape(x, [-1, 8, 4, 1])
        y_image = tf.reshape(y, [-1, 6])
    elif CU_WIDTH==16:
        x_image = tf.reshape(x, [-1, 16, 4, 1])
        y_image = tf.reshape(y, [-1, 6])
    elif CU_WIDTH == 32:
        x_image = tf.reshape(x, [-1, 32, 4, 1])
        y_image = tf.reshape(y, [-1, 6])

    h_cov = sub.overlap_conv(x_image, 3, 3, 1, 16)
    h_condc = res.condc_lumin_4(h_cov)
    y_probabilty = sub.sub_net_8x4_16x4_32x4(h_condc, qp)
    y_predict = tf.argmax(y_probabilty, axis=1)  # 返回最大值索引

    if CU_WIDTH == 8:
        y_one_hot = tf.one_hot(indices=y_predict, depth=2)  # 转换为one—hot vector
        loss_8x4_ce = -tf.reduce_sum(tf.multiply(np.power(p_8x4, adjust_scalar_else).astype(np.float32),tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_8x4, adjust_scalar_else))
        total_loss_8x4 = loss_8x4_ce
        accuracy_8x4 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_8x4)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res6 = tf.trainable_variables(scope='res_unit6')

        return y_probabilty, y_predict, y_one_hot, total_loss_8x4, accuracy_8x4, learning_rate_current, train_step, opt_vars_all, opt_vars_res6

    elif CU_WIDTH == 16:
        y_one_hot = tf.one_hot(indices=y_predict, depth=3)  # 转换为one—hot vector
        loss_16x4_ce = -tf.reduce_sum(tf.multiply(np.power(p_16x4, adjust_scalar_else).astype(np.float32), tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_16x4, adjust_scalar_else))
        total_loss_16x4 = loss_16x4_ce
        accuracy_16x4 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_16x4)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res6 = tf.trainable_variables(scope='res_unit6')

        return y_probabilty, y_predict, y_one_hot, total_loss_16x4, accuracy_16x4, learning_rate_current, train_step, opt_vars_all, opt_vars_res6

    elif CU_WIDTH == 32:
        y_one_hot = tf.one_hot(indices=y_predict, depth=3)  # 转换为one—hot vector
        loss_32x4_ce = -tf.reduce_sum(tf.multiply(np.power(p_32x4, adjust_scalar_else).astype(np.float32), tf.multiply(y_image, tf.log(y_probabilty + 1e-12)))) / np.sum(np.power(p_32x4, adjust_scalar_else))
        total_loss_32x4 = loss_32x4_ce
        accuracy_32x4 = tf.reduce_sum(tf.multiply(y_image, y_one_hot)) / tf.reduce_sum(y_image)
        learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate_current).minimize(total_loss_32x4)
        opt_vars_all = [v for v in tf.trainable_variables()]
        opt_vars_res6 = tf.trainable_variables(scope='res_unit6')

        return y_probabilty, y_predict, y_one_hot, total_loss_32x4, accuracy_32x4, learning_rate_current, train_step, opt_vars_all, opt_vars_res6




