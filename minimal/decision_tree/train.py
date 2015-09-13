# coding: utf-8

import math

from const import CUR_FOLDER, FILE_FOLDER, FILE_TRAIN, FEATURE_NUM, K, VALUES


def get_data():
    data = []
    for i in range(1, K+1):
        filename = CUR_FOLDER + FILE_FOLDER + FILE_TRAIN + '%i.txt' % i
        with open(filename) as f:
            for line in f:
                instance = line.strip().split(',')
                feature = instance[:FEATURE_NUM]
                label = 1 if instance[-1] == 'positive' else 0
                data.append(feature + [label])
    return data


def get_best_feature(data, features=range(FEATURE_NUM)):
    max_info_gain = best_feature = -1
    for feature in features:
        cur_gain = get_info_gain(data, feature)
        if cur_gain > max_info_gain:
            best_feature = feature
            max_info_gain = cur_gain
    return best_feature


def classify(data, feature):
    subsets = []
    for value in VALUES:
        subset = [d for d in data if d[feature] == value]
        subsets.append((value, subset))
    return subsets


def get_info_gain(data, feature):
    entropy = get_entropy(data)
    conditional_entropy = get_conditional_entropy(data, feature)
    return entropy - conditional_entropy


def get_entropy(data):
    len_all = len(data)
    labels = [d[-1] for d in data]
    p_cnt = labels.count(1)
    n_cnt = labels.count(0)
    if not p_cnt or not n_cnt:
        return 0
    p_ratio = float(p_cnt)/len_all
    n_ratio = float(n_cnt)/len_all
    return - (math.log(p_ratio, 2)*p_ratio) - (math.log(n_ratio, 2)*n_ratio)


def get_conditional_entropy(data, feature):
    len_all = len(data) + 0.0
    entropy_all = 0.0
    for value in VALUES:
        subset = [d for d in data if d[feature] == value]
        entropy = get_entropy(subset)
        entropy_all += len(subset) * entropy
    return entropy_all/len_all


class DecisionTree:

    def __init__(self, data):
        self.data = data
        print '\nStart building decision tree:\n'
        self.root = self.build_tree(data[:], range(FEATURE_NUM)[:], {})
        print '\nEnd building decision\n'

    def build_tree(self, data, features, tree):
        print 'Current features: %i\n' % len(features)
        print 'Size of instances to be trained: %i' % len(data)
        if len(features) == 1:
            terminate, combination = self.perfectly_classified(data, features[0])
            if terminate:
                for value, label in combination:
                    tree[value] = {'label': label, 'is_leaf': True}
                return tree
        subtree = {}
        feature = get_best_feature(data[:], features)
        print 'Choose best feature: %i\n' % feature
        features.remove(feature)
        subsets = classify(data[:], feature)
        for value, subdata in subsets:
            if not subdata:
                subtree[value] = {'label': self.assign_label(data[:]),
                                  'is_leaf': True}
            else:
                subsubtree = self.build_tree(subdata, features[:], {})
                subtree[value] = {'subtree': subsubtree, 'is_leaf': False}
        tree[feature] = subtree
        return tree

    def assign_label(self, data):
        """assign a label to a subtree where the branch is empty
        """
        labels = [d[-1] for d in data]
        p_num = labels.count(1)
        n_num = labels.count(0)
        if p_num >= n_num:
            return 1
        else:
            return 0

    def perfectly_classified(self, data, feature):
        """terminate if this feature can uniquely define the data
        which means we can get less than 3 (feature, label) combinations
        """
        combination = set([(d[feature], d[-1]) for d in data])
        if len(combination) < 3:
            return True, combination
        else:
            return False, combination

    def get_label(self, instance):
        tree = self.root.copy()
        features = instance[:FEATURE_NUM]
        while tree:
            for feature, subtree in tree.iteritems():
                value = features[feature]
                tree = subtree[value]
                if tree['is_leaf']:
                    return tree['label']
                tree = tree['subtree']

    def get_trained_labels(self, test_data):
        trained_labels = []
        for instance in test_data:
            trained_labels.append(self.get_label(instance))
        return trained_labels


def train_decision_tree():
    data = get_data()
    return DecisionTree(data)

if __name__ == "__main__":
    train_decision_tree()
