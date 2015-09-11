# coding: utf-8

from collections import defaultdict
import math

import numpy

FILE_FOLDER = '../tic-tac-toe'
FILE_TRAIN = 'tic-tac-toe-train-'
FEATURE_NUM = 9
FEATURE = {'b': 0, 'x': 1, 'o': 2}


def get_data():
    data_all = dict()
    for i in range(1, 7):
        data = []
        with open(FILE_FOLDER + FILE_TRAIN + '%i.txt' % (i)) as f:
            for line in f:
                instance = line.strip().split(',')
                feature = instance[:9]
                label = instance[-1]
                data.append((feature, label))
        data_all[i] = data
    return data_all


class LeafNode:
    """class LeafNode: the leaf node of decision tree
    """

    def __init__(self, label):
        pass


class AttriNode:
    """class AttriNode: the node of desicion tree
    """

    def __init__(self, attr, data=[], leaf=None):
        """attr: dict() -> a dict constains the values of this attribute
           data: list() -> current list of data to be labeled
        """
        self.is_leaf = is_leaf
        self.attr = defaultdict()
        for value in attr:
            self.attr[value]
        self.data = data
        self.next = leaf
        # self.


class DecisionTree:

    def __init__(self, root, attrs, filename, depth=K):
        self.root = set_root()
        self.depth = depth
        self.attrs = attrs
        self.trained_attr_num = 0
        self.entropy = get_entropy(labels)
        self.data = parse_data(filename)

    def parse_data(self, filename):
        """read data from filename
        """
        with open(filename) as f:
            for line in f:
                pass
        return data

    def get_entropy(self, attr, data):
        return entropy

    def get_info_gain(self, attr, data):
        return info_gain

    def set_root(self, attr):
        pass

    def build_tree(self, attrs, data):
        root_attr = None
        max_info_gain = 0
        for attr in attrs:
            info_gain = self.get_info_gain(attr, data)
            if info_gain > max_info_gain:
                root_attr = attr

        # if the current attribute can acurately split the current data, stop
        if max_info_gain == 1:
            pass
        attrs.remove(attr)

        for value in attr:
            self.build_tree(attrs)

    def is_completely_trained(self):
        return len(self.attrs) == self.trained_attr_num

    def get_label(self, instance):
        return label
