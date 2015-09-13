# coding: utf-8

from operator import xor

from const import CUR_FOLDER, FILE_FOLDER, FILE_TEST, FEATURE_NUM
from train import train_decision_tree


def get_data():
    data = []
    labels = []
    filename = CUR_FOLDER + FILE_FOLDER + FILE_TEST
    with open(filename) as f:
        for line in f:
            instance = line.strip().split(',')
            feature = instance[:FEATURE_NUM]
            label = 1 if instance[-1] == 'positive' else 0
            data.append(feature)
            labels.append(label)
    return data, labels


def meature(test_labels, trained_labels):
    labels = zip(test_labels, trained_labels)
    result = [xor(label[0], label[1]) for label in labels]
    return 1 - float(result.count(True))/len(result)


def test():
    test_data, test_labels = get_data()
    tree = train_decision_tree()
    trained_labels = tree.get_trained_labels(test_data)
    score = meature(test_labels, trained_labels)
    print 'precison: ', score

if __name__ == "__main__":
    test()
