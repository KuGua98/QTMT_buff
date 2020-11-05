import tensorflow as tf
from activate import activate
from tensorflow.contrib.layers import xavier_initializer


# half mask operation
keep_prob_qp_half = 0.5

# 权重和偏置初始化方式
initializer_oc = xavier_initializer()
initializer_subnet_weight = xavier_initializer()
initializer_subnet_biases = xavier_initializer()


# qp half mask
def qp_half_mask(x, qp):
    x1 = tf.nn.dropout(x, keep_prob_qp_half)
    x2 = x - x1
    # qp正规化
    qp = qp/51
    qp = tf.expand_dims(qp, 2)
    qp = tf.expand_dims(qp, 2)
    # half mask
    h = x1*qp + x2

    # x1 = tf.nn.dropout(x, keep_prob_qp_half)
    # h = x1*qp

    return h

# 重叠卷积层
def overlap_conv(x, k_width, k_height, num_fileters_in, num_filters_out):
    # 激活函数采用Prelu
    acti_mode = 2

    weight_name = 'w_oc'
    biases_name = 'b_oc'
    with tf.variable_scope('overlab_conv', reuse=tf.AUTO_REUSE):
        w_oc = tf.get_variable(weight_name, [k_width, k_height,num_fileters_in, num_filters_out],initializer=initializer_oc)
        b_oc = tf.get_variable(biases_name, [num_filters_out], initializer=initializer_oc)
    h_oc = tf.nn.conv2d(x, w_oc, strides=[1, 1, 1, 1], padding='SAME')
    h_f = activate(h_oc+b_oc, acti_mode)

    return h_f


# 非重叠卷积层
def non_overlap_conv(x, k_width, k_height, num_filters_in, num_filters_out, CU_NAME, nc_index):
    # 激活函数采用Prelu
    acti_mode = 2

    weights_name = 'w_subnet_nc_' + CU_NAME + '_' + str(nc_index)
    biases_name = 'b_subnet_nc_' + CU_NAME + '_' + str(nc_index)
    with tf.variable_scope('non_overlap_conv', reuse=tf.AUTO_REUSE):
        w_nc = tf.get_variable(weights_name, [k_width, k_height, num_filters_in, num_filters_out],initializer=initializer_subnet_weight)
        b_nc = tf.get_variable(biases_name, [num_filters_out], initializer=initializer_subnet_biases)

    h_nc = tf.nn.conv2d(x, w_nc, strides=[1, k_width, k_height, 1], padding='SAME')
    h_nc = activate(h_nc + b_nc, acti_mode)
    # print(h_nc)
    return(h_nc)


# 全连接层
def full_connect(x, num_filters_in, num_filters_out, acti_mode, CU_NAME, fc_index):
    weights_name = 'w_subnet_fc_' + CU_NAME + '_' + str(fc_index)
    biases_name = 'b_subnet_fc_' + CU_NAME + '_' + str(fc_index)

    if fc_index == 1:
        x_flat = tf.reshape(x, [-1, x.shape[1]*x.shape[2]*num_filters_in])
    if fc_index == 2:
        x_flat = x

    with tf.variable_scope('full_connect', reuse=tf.AUTO_REUSE):
        w_fc = tf.get_variable(weights_name, [x_flat.shape[1], num_filters_out], initializer=initializer_subnet_weight)
        b_fc = tf.get_variable(biases_name, [num_filters_out], initializer=initializer_subnet_biases)

    h_fc = tf.matmul(x_flat, w_fc) + b_fc
    h_fc = activate(h_fc, acti_mode)
    # print(h_fc)
    return h_fc


def sub_net_128():
    return


# Sub-network
def sub_net_64(x_input, qp):
    CU_NAME = "64x64"

    x_half1 = qp_half_mask(x_input, qp)
    # non_overlap_conv(x, k_width, k_height, num_filters_in, num_filters_out, cu_size, nc_index):
    h_nc1 = non_overlap_conv(x_half1, 4, 4, 16, 8, CU_NAME, 1)
    h_nc2 = non_overlap_conv(h_nc1, 4, 4, 8, 8, CU_NAME, 2)

    h_half2 = qp_half_mask(h_nc2, qp)
    # full_connect(x, num_filters_in, num_filters_out, acti_mode, cu_size, fc_index):
    h_f1 = full_connect(h_half2, 8, 8, 2, CU_NAME, 1)
    h_f2 = full_connect(h_f1, 8, 2, 3, CU_NAME, 2)    # 64*64 只能进行四叉树分割？？？
    return h_f2


def sub_net_32(x_input, qp, cu_width=32, cu_height=32):
    CU_NAME = "32x32"

    x_half1 = qp_half_mask(x_input, qp)
    h_nc1 = non_overlap_conv(x_half1, cu_width/8, cu_height/8, 16, 16, CU_NAME, 1)
    h_nc2 = non_overlap_conv(h_nc1, 4, 4, 16, 32, CU_NAME, 2)
    h_nc3 = non_overlap_conv(h_nc2, 2, 2, 32, 128, CU_NAME, 3)

    h_half2 = qp_half_mask(h_nc3, qp)
    h_f1 = full_connect(h_half2, 128, 64, 2, CU_NAME, 1)
    h_f2 = full_connect(h_f1, 64, 6, 3, CU_NAME, 2)
    return h_f2


def sub_net_16(x_input, cu_width, cu_height, qp):
    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    x_half1 = qp_half_mask(x_input, qp)

    h_nc1 = non_overlap_conv(x_half1, cu_width/4, cu_height/4, 16, 16, CU_NAME, 1)

    h_nc2 = non_overlap_conv(h_nc1, 2, 2, 16, 32, CU_NAME, 2)
    h_nc3 = non_overlap_conv(h_nc2, 2, 2, 32, 64, CU_NAME, 3)
    h_half2 = qp_half_mask(h_nc3, qp)
    h_f1 = full_connect(h_half2, 64, 64, 2, CU_NAME, 1)

    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    if cu_width==16:
        h_f2 = full_connect(h_f1, 64, 6, 3, CU_NAME, 2)
    elif cu_width==32:
        h_f2 = full_connect(h_f1, 64, 6, 3, CU_NAME, 2)

    return h_f2


def sub_net_8(x_input, cu_width, cu_height, qp):
    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    x_half1 = qp_half_mask(x_input, qp)
    h_nc1 = non_overlap_conv(x_half1, cu_width/2, cu_height/2, 16, 16, CU_NAME, 1)

    CU_NAME = str(cu_height)
    h_nc2 = non_overlap_conv(h_nc1, 2, 2, 16, 32, CU_NAME, 2)
    h_half2 = qp_half_mask(h_nc2, qp)
    h_f1 = full_connect(h_half2, 32, 16, 2, CU_NAME, 1)

    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    if cu_width==8:
        h_f2 = full_connect(h_f1, 16, 3, 3, CU_NAME, 2)
    elif cu_width==16:
        h_f2 = full_connect(h_f1, 16, 4, 3, CU_NAME, 2)
    elif cu_width == 32:
        h_f2 = full_connect(h_f1, 16, 4, 3, CU_NAME, 2)

    return h_f2


def sub_net_4(x_input, cu_width, cu_height, qp):
    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    x_half1 = qp_half_mask(x_input, qp)
    h_nc1 = non_overlap_conv(x_half1, cu_width/2, cu_width/2, 16, 16, CU_NAME, 1)

    CU_NAME = str(cu_height)
    h_nc2 = non_overlap_conv(h_nc1, 2, 2, 16, 32, CU_NAME, 2)
    h_half2 = qp_half_mask(h_nc2, qp)
    h_f1 = full_connect(h_half2, 32, 16, 2, CU_NAME, 1)

    CU_NAME = str(cu_width) + 'x' + str(cu_height)
    if cu_width==8:
        h_f2 = full_connect(h_f1, 16, 2, 3, CU_NAME, 2)
    elif cu_width==16:
        h_f2 = full_connect(h_f1, 16, 3, 3, CU_NAME, 2)
    elif cu_width == 32:
        h_f2 = full_connect(h_f1, 16, 6, 3, CU_NAME, 2)

    return h_f2