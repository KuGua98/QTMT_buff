import tensorflow as tf
import tflearn
from activate import activate
from tensorflow.contrib.layers import xavier_initializer
import math


# x = tf.placeholder(tf.float32, [None,784])
# y = tf.placeholder(tf.float32, [None, 10])
# sess = tf.InteractiveSession()

# 权重和偏置初始化方式
initializer_Res_weight = xavier_initializer()
initializer_Res_biases = xavier_initializer()

# 激活方式
acti_mode = 2

#在这里定义残差网络的id_block块，此时输入和输出维度相同
# kernel_size=3   in_filter = out_filters = 16  stage = k
def identity_block(X_input, kernel_size, in_filter, out_filters, k):
        """
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        k -- index of the residual units

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res_unit' + str(k)

        # 卷积块中的参数命名
        weights_name_con1 = 'w_res' + str(k) + '_1'
        weights_name_con2 = 'w_res' + str(k) + '_2'

        f1 = out_filters    # 二卷积
        f2 = out_filters
        with tf.variable_scope(block_name, reuse=tf.AUTO_REUSE):
            X_shortcut = X_input

            #first
            W_conv1 = tf.get_variable(weights_name_con1, [kernel_size, kernel_size, in_filter, f1], initializer=initializer_Res_weight)
            X1 = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X1 = tf.layers.batch_normalization(X1, training=True)
            # b_conv1 = tf.get_variable(biases_name_con1, [f1], initializer=initializer_Res_biases)
            # X1 = activate(X1 + b_conv1, acti_mode)
            X1 = activate(X1, acti_mode)

            #second
            W_conv2 = tf.get_variable(weights_name_con2, [kernel_size, kernel_size, f1, f2], initializer=initializer_Res_weight)
            X2 = tf.nn.conv2d(X1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X2 = tf.layers.batch_normalization(X2, training=True)
            # b_conv2 = tf.get_variable(biases_name_con2, [f2], initializer=initializer_Res_biases)
            # X2 = X2 + b_conv2
            # X = activate(X + b_conv2, acti_mode)

            #final step
            add = tf.add(X2, X_shortcut)
            # b_conv_fin = tf.get_variable(biases_name_fin, [f2], initializer=initializer_Res_biases)
            # add_result = activate(add + b_conv_fin, acti_mode)
            add_result = activate(add, acti_mode)

        return add_result


def identity_block_reuse(X_input, kernel_size, in_filter, out_filters, k):

    block_name = 'res_unit' + str(k)
    weights_name_con1 = 'w_res' + str(k) + '_1'
    weights_name_con2 = 'w_res' + str(k) + '_2'

    f1 = out_filters  # 二卷积
    f2 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 = tf.get_variable(weights_name_con1, [kernel_size, kernel_size, in_filter, f1],initializer=initializer_Res_weight)
        X1 = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X1 = tf.layers.batch_normalization(X1, training=True)
        X1 = activate(X1, acti_mode)

        # second
        W_conv2 = tf.get_variable(weights_name_con2, [kernel_size, kernel_size, f1, f2],initializer=initializer_Res_weight)
        X2 = tf.nn.conv2d(X1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X2 = tf.layers.batch_normalization(X2, training=True)

        # final step
        add = tf.add(X2, X_shortcut)
        add_result = activate(add, acti_mode)

    return add_result

# 训练和预测时的条件卷积不一样，需要分开写
# def cond_conv_64(x_input, cu_size):
    # k = math.log(256/cu_size, 2)
    # h = identity_block(x_input, 3, 16, 16, k)
    # return

# 亮度分量的条件卷积   【注意】 色度分量的需要另外定义
def condc_lumin_64(x_input):
    h = identity_block(x_input, 3, 16, 16, 1)
    h = identity_block(h, 3, 16, 16, 2)
    return h

def condc_lumin_32(x_input):
    h = identity_block(x_input, 3, 16, 16, 3)
    return h

def condc_lumin_16(x_input):
    cu_width = x_input.shape[1].value
    cu_height = x_input.shape[2].value
    # h = identity_block(x_input, 3, 16, 16, 4)
    if cu_width == cu_height:
        h = identity_block(x_input, 3, 16, 16, 4)
    else:
        h = identity_block_reuse(x_input, 3, 16, 16, 4)
    return h

def condc_lumin_8(x_input):
    cu_width = x_input.shape[1].value
    cu_height = x_input.shape[2].value
    # h = identity_block(x_input, 3, 16, 16, 5)
    if cu_width == cu_height:
        h = identity_block(x_input, 3, 16, 16, 5)
    else:
        h = identity_block_reuse(x_input, 3, 16, 16, 5)
    return h

def condc_lumin_4(x_input):
    cu_width = x_input.shape[1].value
    cu_height = x_input.shape[2].value
    # h = identity_block(x_input, 3, 16, 16, 5)
    if cu_width ==2* cu_height:
        h = identity_block(x_input, 3, 16, 16, 6)
    else:
        h = identity_block_reuse(x_input, 3, 16, 16, 6)
    # h = identity_block(x_input, 3, 16, 16, 6)
    return h