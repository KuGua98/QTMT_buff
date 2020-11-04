import sys
import h5py
import time
import numpy as np

NAME_LIST = ['64x64','32x32','16x16','32x16','8x8','32x8','16x8','8x4','32x4','16x4']
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]

start = time.time()

h5f = h5py.File('Samples_train.h5', 'r')
for name in NAME_LIST:
    data_Y = h5f[name]
    print(len(data_Y))
h5f.close()

end = time.time()


print(start)
print(end)
print("time is: ", end-start, "s")