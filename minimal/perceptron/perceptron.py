# coding: utf-8

from random import shuffle, uniform
# import IPython

from const import CUR_FOLDER, FILE_FOLDER, FILE_TRAIN, FILE_TEST, FILE_NUM, \
    FEATURE_NUM, MU, R, EPOCH


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
                feature = [0 for _ in range(j+2)]
                for x in line.split()[1:]:
                    [key, val] = x.split(':')
                    feature[int(key)] = float(val)
                    data[file_type].append((label, feature))
    return data


def prepare_weight(j):
    """j -> feature number
    randomly generate weights for initial state
    return a j+1 vector"""
    return [uniform(0, 1) for _ in range(j+2)]


def dot_prod(a, b):
    """a -> row vector
       b -> column vector"""
    return sum([a[i]*b[i] for i in range(len(a))])


def update_weight(weight, feature, r, label):
    for idx, w in enumerate(weight[:]):
        weight[idx] += r*label*feature[idx]


def train_pair(r, j, data, mu=0):
    """mu -> margin
       r -> learning rate
       j -> feature number
    """
    mistake = 0
    # r = cross_validation()
    w = prepare_weight(j)
    for label, feature in data:
        if label*(dot_prod(w, feature)) <= mu:
            mistake += 1
            update_weight(weight=w, r=r, feature=feature, label=label)
    return w


def test_pair(w, r, j, data, mu=0):
    """mu -> margin
       w -> weight
       r -> learning rate
       j -> feature number
    """
    mistake = 0
    for label, feature in data:
        if label*(dot_prod(w, feature)) <= mu:
            mistake += 1
    return 1 - float(mistake)/len(data)


def train_batch(data_all, epoch=EPOCH, mu_list=MU, r_list=R):
    for i in FILE_NUM:
        for j in FEATURE_NUM:
            data = data_all[(i, j)]
            train_data = data['train']
            for r in r_list:
                w = train_pair(data=data['train'], j=j, r=r)
                for _ in range(epoch):
                    shuffle(train_data)
                    w = verctor_add(w, train_pair(data=train_data, j=j, r=r))
                w = [_w/epoch for _w in w]
                precision = test_pair(data=data['test'], j=j, r=r, w=w)
                print 'data %i, feature number %i, learning rate %f, precision %f' \
                    % (i, j, r, precision)

            # try different mu
            for mu in mu_list:
                w = train_pair(data=data['train'], j=j, r=1)
                for _ in range(epoch):
                    shuffle(train_data)
                    w = verctor_add(w, train_pair(mu=mu, data=train_data, j=j,
                                                  r=1))
                w = [_w/epoch for _w in w]
                precision = test_pair(mu=mu, data=data['test'], j=j, r=1, w=w)
                print 'data %i, feature number %i, mu %f, precision %f' \
                    % (i, j, mu, precision)


def verctor_add(w, w_):
    return [w[i]+w_[i] for i in range(len(w))]


def train_online(data_all):
    for i in FILE_NUM:
        for j in FEATURE_NUM:
            data = data_all[(i, j)]
            w = train_pair(data=data['train'], j=j, r=1)
            precision = test_pair(data=data['test'], j=j, r=1, w=w)
            print 'data %i, feature number %i, precision %f' \
                % (i, j, precision)
            w = train_pair(mu=0.3, data=data['train'], j=j, r=1)
            precision = test_pair(data=data['test'], j=j, r=1, w=w)
            print 'data %i, feature number %i, mu %f, precision %f' \
                % (i, j, 0.3, precision)


def train():
    data = prepare_data()
    train_batch(data)

if __name__ == "__main__":
    train()
