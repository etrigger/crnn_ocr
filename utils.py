import os
import glob
# import random
import cv2
import numpy as np
import tensorflow as tf
# path = '.\part2\\'


def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def get_char_res(char_dict, decoded):

    res = []
    for i in decoded:
        char = char_dict[i]
        res.append(char)
    return res


def get_char_dict(path):
    char_dict = []
    txt_files = glob.glob(path + '*.txt')
    # print(len(txt_files))
    for file in txt_files:
        with open(file, 'r') as f:
            text = f.readline()
        char_dict += text
    char_dict = set(char_dict)
    char_dict = list(char_dict)
    dict_size = len(char_dict)
    print("dict_size:{}".format(dict_size))
    return char_dict


def get_data(path, char_dict):
    train_x = []
    train_y = []
    txt_files = glob.glob(path + '*.txt')
    # random.shuffle(txt_files)
    for file in txt_files:
        base_name = os.path.basename(file)
        file_name, _ = os.path.splitext(base_name)
        image = cv2.imread(path + file_name + '.jpg', cv2.IMREAD_GRAYSCALE)
        train_x.append(image)
        with open(file, 'r') as f:
            label = []
            text = f.readline()
            for c in text:
                index = char_dict.index(c)
                label.append(index)
        train_y.append(label)
    for i, img in enumerate(train_x):
        if train_x[i] is None:
            del train_x[i]
            del train_y[i]
    print("train_size:{}".format(len(train_x)))
    return train_x, train_y


