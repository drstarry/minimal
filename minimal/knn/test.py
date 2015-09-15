# coding: utf-8

from operator import xor
import time

from const import CUR_FOLDER, FILE_FOLDER, FILE_TEST, FILE_TRAIN, FEATURE_NUM, FOLD
from train import crossvalidation, knn, measure


def get_test_data():
    test_data = []
    labels = []
    filename = CUR_FOLDER + FILE_FOLDER + FILE_TEST
    with open(filename) as f:
        for line in f:
            instance = line.strip().split(',')
            feature = instance[:FEATURE_NUM]
            label = 1 if instance[-1] == 'positive' else 0
            test_data.append(feature)
            labels.append(label)
    return test_data, labels


def get_training_data():
    train_data = []
    for i in range(1, FOLD+1):
        filename = CUR_FOLDER + FILE_FOLDER + FILE_TRAIN + '%i.txt' % i
        with open(filename) as f:
            for line in f:
                instance = line.strip().split(',')
                feature = instance[:FEATURE_NUM]
                label = 1 if instance[-1] == 'positive' else 0
                train_data.append(feature + [label])
    return train_data


def test():
    print 'begin training...'
    start = time.time()
    test_data, test_labels = get_test_data()
    train_data = get_training_data()
    k = crossvalidation()
    print 'end training...'
    print 'start test...'
    trained_labels = knn(train_data, test_data, k)
    score = measure(test_labels, trained_labels)
    end = time.time()
    print 'end test...'
    print 'training time (s): ', end - start
    print 'precison: ', score

if __name__ == "__main__":
    test()
