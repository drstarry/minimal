# coding: utf-8

import math

from const import CUR_FOLDER, FILE_FOLDER, FILE_TRAIN,  FEATURE_NUM, K, FOLD, VALUES


def get_data(test_no):
    train_data = []
    test_data = []
    for i in range(1, FOLD+1):
        filename = CUR_FOLDER + FILE_FOLDER + FILE_TRAIN + '%i.txt' % i
        with open(filename) as f:
            for line in f:
                instance = line.strip().split(',')
                feature = instance[:FEATURE_NUM]
                label = 1 if instance[-1] == 'positive' else 0
                if i-1 == test_no:
                    test_data.append(feature + [label])
                else:
                    train_data.append(feature + [label])
    return train_data, test_data


def crossvalidation():
    print 'begin crossvalidation'
    k_measure = []
    for k in K:
        print 'current k: ', k
        precision = 0.0
        for i in range(FOLD):
            train_data, test_data = get_data(test_no=i)
            trained_labels = knn(train_data, test_data, k)
            test_labels = [d[-1] for d in test_data]
            _precision = measure(test_labels, trained_labels)
            precision += _precision
        avg_p = precision/FOLD
        k_measure.append((k, avg_p))
        print 'avg precision: ', avg_p
    k_measure.sort(key=lambda x: x[1], reverse=True)
    print 'best k trained: ', k_measure[0][0]
    print 'best avg precision: ', k_measure[0][1]
    return k_measure[0][0]


def measure(test_labels, trained_labels):
    return 1 - float(hamming_distance(trained_labels, test_labels))/len(test_labels)


def knn(train_data, test_data, k):
    labels = []
    for test in test_data:
        distance = []
        for train in train_data:
            dis = hamming_distance(train[:FEATURE_NUM], test[:FEATURE_NUM])
            distance.append((dis, train[-1]))
        distance.sort(key=lambda x: x[0])
        label_k = [d[1] for d in distance[:k]]
        label = 1 if label_k.count(1) >= label_k.count(0) else 0
        labels.append(label)
    return labels


def XOR(x, y):
    if x != y:
        return True
    else:
        return False


def hamming_distance(x, y):
    both = zip(x, y)
    return [XOR(_x, _y) for [_x, _y] in both].count(True)


if __name__ == '__main__':
    crossvalidation()
