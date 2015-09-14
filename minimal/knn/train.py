# coding: utf-8

import math

from const import CUR_FOLDER, FILE_FOLDER, FILE_TRAIN, FEATURE_NUM, K, VALUES


def get_data():
    data_all = []
    for i in range(1, K+1):
        data = []
        filename = CUR_FOLDER + FILE_FOLDER + FILE_TRAIN + '%i.txt' % i
        with open(filename) as f:
            for line in f:
                instance = line.strip().split(',')
                feature = instance[:FEATURE_NUM]
                label = 1 if instance[-1] == 'positive' else 0
                data.append(feature + [label])
        data_all.append(data)
    return data_all
