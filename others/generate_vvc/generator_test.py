import numpy as np
import tensorflow as tf

def data_generator():
    dataset = np.array(range(5))
    index =0
    while True:
        d = dataset[index]
        yield d
        index += 1
        if index == 5:
            index=0

    # for d in dataset:
    #     yield d
    #     if d == 4:
    #         d=0


dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32), (tf.TensorShape([])))
dataset = dataset.shuffle(2)

dataset = dataset.batch(2)
dataset = dataset.repeat(2)




iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

with tf.Session() as sess:
    try:
        batch_num = 0
        while True:
            one_batch = sess.run(one_element)
            print('Batch No. %d:' % batch_num)
            print(one_batch)
            print('')
            batch_num += 1

    except tf.errors.OutOfRangeError:
        print('end!')



# dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5, 6, 7]))
# 对数据打乱，设置缓冲区大小为4
# dataset = dataset.shuffle(4)
# for ele in dataset:
#     print(ele.numpy())
# 对数据进行重复，每一次都是乱序的，参数为重复的次数，如果不写默认无限次重复
# dataset = dataset.repeat(count=3)
# for ele in dataset:
#     print(ele.numpy())
# 取出batch_size大小的数据
# dataset = dataset.batch(3)
# for ele in dataset:
#    print(ele.numpy())