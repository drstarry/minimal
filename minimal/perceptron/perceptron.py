# coding: utf-8

from random import shuffle, uniform
from collections import defaultdict
# import IPython

from const import CUR_FOLDER, FILE_FOLDER, FILE_TRAIN, FILE_TEST, FILE_NUM, \
    FEATURE_NUM, MU, R, EPOCH


def prepare_data_try():
    data = []
    with open('data') as f:
        for line in f:
            label = int(line.split()[0])
            feature = [0 for _ in range(5)]
            for x in line.split()[1:]:
                [key, val] = x.split(':')
                feature[int(key)] = float(val)
            data.append((label, feature))
    return data


def prepare_data():
    data = dict()
    for i in FILE_NUM:
        for j in FEATURE_NUM:
            data[(i, j)] = prepare_pair(i, j)
    return data


def prepare_pair(i, j):
    """i -> data NO. (data0, data1)
       j -> feature number
    """
    data = dict()
    for FILE in [FILE_TRAIN, FILE_TEST]:
        file_type = 'train' if FILE == FILE_TRAIN else 'test'
        data[file_type] = []
        file_name = CUR_FOLDER + FILE_FOLDER % i + FILE % (i, j)
        with open(file_name) as f:
            for line in f:
                label = int(line.split()[0])
                feature = [0 for _ in range(j+1)]
                for x in line.split()[1:]:
                    [key, val] = x.split(':')
                    feature[int(key)] = float(val)
                data[file_type].append((label, feature))
    return data


def prepare_weight(j):
    """j -> feature number
    randomly generate weights for initial state
    return a j+1 vector"""
    return [uniform(0, 1) for _ in range(j+1)]


def dot_prod(a, b):
    """a -> row vector
       b -> column vector"""
    return sum([a[i]*b[i] for i in range(len(a))])


def update_weight(weight, feature, r, label):
    for idx, w in enumerate(weight[:]):
        weight[idx] += r*label*feature[idx]


def verctor_add(w, w_):
    return [w[i]+w_[i] for i in range(len(w))]


def train_pair(j, data, w, r=1, mu=0, agg=False):
    """mu -> margin
       r -> learning rate
       j -> feature number
       agg -> if this is a aggressive perceptron
    """
    mistake = 0
    a = w
    for label, feature in data:
        if label*(dot_prod(w, feature)) <= mu:
            mistake += 1
            if agg:
                rate = learning_rate(mu=mu, w=w, label=label, feature=feature)
            else:
                rate = r
            update_weight(weight=w, r=rate, feature=feature, label=label)
            a = verctor_add(a, w)
    return a, mistake


def test_pair(w, j, data, mu=0):
    """mu -> margin
       w -> weight
       j -> feature number
    """
    mistake = 0
    for label, feature in data:
        if label*(dot_prod(w, feature)) <= mu:
            mistake += 1
    return 1 - float(mistake)/len(data)


def learning_rate(mu, w, label, feature):
    return (mu-label*(dot_prod(w, feature)))/(dot_prod(feature, feature)+1)


def train_batch(data_all, mu, r, epoch=EPOCH, agg=False):
    for i in FILE_NUM:
        for j in FEATURE_NUM:
            data = data_all[(i, j)]
            train_data = data['train']
            w = prepare_weight(j)
            mistake_all = 0.0
            for _ in range(epoch):
                shuffle(train_data)
                _w, mistake = train_pair(mu=mu, data=train_data, j=j,
                                         r=1, w=w, agg=agg)
                mistake_all += mistake
                w = verctor_add(w, _w)
            mistake_all /= epoch
            precision = test_pair(mu=mu, data=data['test'], j=j, w=w)
            print 'data %i, feature number %i, mu %f, train mistake %i,' \
                  ', precision %f' \
                % (i, j, mu, mistake_all, precision)


def split_data(data):
    """5 fold cross validation
    """
    result = []
    SIZE = len(data)/5
    for i in range(5):
        test = data[SIZE*i:SIZE*(i+1)]
        train = [x for x in data if x not in test]
        result.append((train, test))
    return result


def cv_simple(data, j=10, i=0, agg=False):
    result = split_data(data)
    precision_best = r_best = 0
    for r in R:
        precision = precision_best = 0
        for train, test in result:
            w = prepare_weight(j=10)
            w, mistake_train = train_pair(data=train, r=r,
                                          w=w, j=j, agg=agg)
            precision += test_pair(data=test, j=j, w=w)
        precision /= 5
        if precision > precision_best:
            precision = precision_best
            r_best = r
    return r_best


def cv_margin(data, j=10, i=0, agg=False):
    result = split_data(data)
    precision_best = mu_best = r_best = 0
    for r in R:
        for mu in MU:
            precision = precision_best = 0
            for train, test in result:
                w = prepare_weight(j=10)
                w, mistake_train = train_pair(data=train, r=r, mu=mu,
                                              w=w, j=j, agg=agg)
                precision += test_pair(data=test, j=j, w=w)
            precision /= 5
        if precision > precision_best:
            precision = precision_best
            r_best = r
            mu_best = mu
    return r_best, mu_best


def exp_2(data_all, j=10, i=0, agg=False):
    """play with data 0 dimension 10
    """
    data = data_all[(i, j)]
    result = defaultdict(list)

    r = cv_simple(data['train'])
    print 'best learning rate: ', r
    w = prepare_weight(j=j)
    w, mistake_train = train_pair(data=data['train'], r=r,
                                  w=w, j=j, agg=agg)
    precision = test_pair(data=data['test'], j=j, w=w)
    result['simple'].append((r, mistake_train, precision))
    print 'data %i, feature number %i, learning rate %f,' \
          ' update times %i, precision %f' \
        % (i, j, r, mistake_train, precision)

    r, mu = cv_margin(data['train'])
    print 'best learning rate and margin: ', r, mu
    w = prepare_weight(j=j)
    w, mistake_train = train_pair(mu=mu, data=data['train'], r=r, j=j, w=w, agg=agg)
    precision = test_pair(data=data['test'], j=j, w=w)
    result['margin'].append((r, mu, mistake_train, precision))
    print 'data %i, feature number %i, learning rate %f, mu %f, ' \
          'update times %i, precision %f' % (i, j, r, mu, mistake_train, precision)
    return result


def exp_1(data, agg=False):
    weight, mistake_train = train_pair(data=data, j=4, w=prepare_weight(4),
                                       agg=agg)
    print 'weight(first item is bias): ', weight
    print 'number of mistake: ', mistake_train


def train():
    data = prepare_data()
    data_try = prepare_data_try()

    # fixed learning rate:
    exp_1(data_try)
    # exp_2(data)
    # result = train_batch(data, mu=0.01, r=1)

    # aggressive learning rate
    # exp_1(data_try, agg=True)
    # exp_2(data, agg=True)
    # train_batch(data, agg=True)

if __name__ == "__main__":
    train()
