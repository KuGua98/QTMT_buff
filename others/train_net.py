import tensorflow as tf
from others import input_data as input_data
import net_cu as net

MINI_BATCH_SIZE = 32
ITER_TIMES = 500000
NUM_CHANNELS = 1

CU_WIDTH = 64
CU_HEIGHT = 64

# IMAGE_SIZE = [[64,64],[32,32],[16,16],[32,16],[8,8],[32,8],[16,8],[8,4],[32,4],[16,4]]
# NUM_SAMPLE_LENGTH = [4106, 1034, 266, 522, 74, 266, 138, 42, 138, 74]
# IS_RELOAD = False # whether to reload the model trained last time. False: train from scretch, True: fine-tune

LEARNING_RATE_INIT = 1e-4
DECAY_RATE = 0.01
DECAY_STEP = 2000

x = tf.placeholder("float", [None, CU_WIDTH, CU_HEIGHT , NUM_CHANNELS])
y = tf.placeholder("float", [None, 1])
qp = tf.placeholer("float", [None, 1])
global_step = tf.placeholder("float")
# rd_cost = tf.placeholder("float", [None, 6])
# rd_cost_min = tf.placeholder("float", [None, 1])

y_probabilty, y_predict, y_one_hot, total_loss_64x64, loss_64x64_list, learning_rate_current, train_step, accuracy_64x64_list, opt_vars_all \
     = net.net_64x64(x, y, qp, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)
# y_probabilty, y_predict, y_one_hot, total_loss_64x64, loss_64x64_list, learning_rate_current, train_step, accuracy_64x64_list, opt_vars_all \
#     = net.net_64x64(x, y, rd_cost, rd_cost_min, qp, global_step, LEARNING_RATE_INIT, DECAY_RATE, DECAY_STEP)

data_sets = input_data.read_data_set()
saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)
sess = tf.Session()

train_accuracy_list = list()
train_loss_list = list()
train_tendency_list = list()
step_list = list()

sess.run(tf.global_variables_initializer())

# if IS_RELOAD == True:
# 	saver.restore(sess, 'Models/model.dat')
# 	iter_times_last = reload_loss_and_accuracy()
# else:
# 	iter_times_last = 0
# print('iter_times_last = %d' % iter_times_last)
# if IS_RELOAD == False:
# 	evaluate_loss_accuracy(iter_times_last, LEARNING_RATE_INIT)

iter_times_last = 0
evaluate_loss_accuracy(iter_times_last, LEARNING_RATE_INIT)

for i in range(ITER_TIMES):
    step = i + iter_times_last + 1
    batch = data_sets.train.next_batch_random(MINI_BATCH_SIZE)

    feed_step = step
    sess.run([learning_rate_current, train_step], feed_dict={x:batch[0], y:batch[1], qp:batch[2], global_step:feed_step})
    # sess.run([learning_rate_current, train_step], feed_dict={x:batch[0], y:batch[1], qp:batch[2], rd_cost:batch[3], rd_cost_min:batch[4]})

    if step % ITER_TIMES_PER_EVALUATE == 0:
        evaluate_loss_accuracy(step, learning_rate_value)