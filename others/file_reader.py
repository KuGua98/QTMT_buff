import os
import numpy as np


def getFileSize(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


class FileReader(object):
    def __init__(self):
        self._file_name = ''
        self._file_pointer = None
        self._current_bytes = 0
        self._max_bytes = 0

    #  max_bytes 可以不要 ？
    def initialize(self, file_name, max_bytes):
        print('Opening File : %s' % file_name)
        self._file_name = file_name
        self._file_pointer = open(file_name, 'rb')
        self._current_bytes = 0

        file_bytes = getFileSize(file_name)
        # self._max_bytes = file_bytes
        if max_bytes < file_bytes:
            self._max_bytes = max_bytes
        else:
            self._max_bytes = file_bytes

    @property
    def get_file_name(self):
        return self._file_name

    @property
    def get_file_pointer(self):
        return self._file_pointer

    @property
    def get_max_bytes(self):
        return self._max_bytes

    @property
    def get_current_bytes(self):
        return self._current_bytes

    def read_data(self, read_bytes, isloop):
        if read_bytes > self._max_bytes:
            read_bytes = self._max_bytes
        if self._current_bytes + read_bytes <= self._max_bytes:
            buf = self._file_pointer.read(read_bytes)
            data = np.frombuffer(buf,dtype=float)
            self._current_bytes += read_bytes
        else:
            if isloop == False:
                self._file_pointer = open(self._file_name, 'rb')
                buf = self._file_pointer.read(read_bytes)
                data = np.frombuffer(buf,dtype=float)
                self._current_bytes = read_bytes
            else:
                buf = self._file_pointer.read(self._max_bytes - self._current_bytes)
                data1 = np.frombuffer(buf,dtype=float)
                self._file_pointer = open(self._file_name, 'rb')
                buf = self._file_pointer.read(read_bytes - (self._max_bytes - self._current_bytes))
                data2 = np.frombuffer(buf,dtype=float)
                data = np.concatenate((data1, data2))
                self._current_bytes = read_bytes - (self._max_bytes - self._current_bytes)
        return data
