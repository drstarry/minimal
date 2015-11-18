# coding: utf-8

from random import shuffle
from collections import defaultdict

from config import CUR_FOLDER, FILE_FOLDER, FILE_TRAIN, FOLD, C, \
    RHO_0, EPOCH_CV, EPOCH, ORIGIN, SCALED, FILE_TEST
from util import verctor_add, dot_prod, scale_vector, chunks


def prepare_data(FILE_NAME):
    with open(CUR_FOLDER+FILE_FOLDER+FILE_NAME) as f:
        for line in f:
            label = int(line.split()[0])
            if label == 0:
                label = -1
            feature = [0 for _ in line.split()]
            feature[0] = 1
            for x in line.split()[1:]:
                [key, val] = x.split(':')
                feature[int(key)] = float(val)
            yield (label, feature)


def tranpose_data(FILE_NAME):
    with open(CUR_FOLDER+FILE_FOLDER+FILE_NAME) as f:
        for line in f:
            label = int(line.split()[0])
            if label == 0:
                label = -1
            feature = [1]
            for x in line.split()[1:]:
                [key1, val1] = x.split(':')
                for y in line.split()[1:]:
                    [key2, val2] = y.split(':')
                    if int(key1) <= int(key2):
                        feature.append(float(val1)*float(val2))
            yield (label, feature)


def get_furthest(data):
    max_dis = float('-inf')
    for y_i, x_i in data:
        dis = pow(dot_prod(x_i[1:], x_i[1:]), 0.5)
        max_dis = max(max_dis, dis)
    return max_dis


class SVM(object):

    def __init__(self, data, C, RHO_O):
        self.data = data
        self.C = C
        self.RHO_0 = RHO_O

    def margin(self, data, weight):
        posi_margin = float('inf')
        nege_margin = float('-inf')
        for y_i, x_i in data:
            if y_i*dot_prod(weight, x_i) > 1:
                dis = dot_prod(weight, x_i) / pow(dot_prod(weight, weight), 0.5)
                if y_i == -1:
                    nege_margin = max(nege_margin, dis)
                else:
                    posi_margin = min(posi_margin, dis)
        return posi_margin, nege_margin

    def test(self, data, weight):
        mistake = 0
        for y_i, x_i in data:
            if y_i*dot_prod(weight, x_i) <= 1:
                mistake += 1
        return 1 - float(mistake)/len(data)

    def cross_validation(self, fold, epoch):
        print 'doing cross validation...'
        splited_data = list(chunks(self.data, fold))
        hyper_test = defaultdict(int)
        for idx, (train, test) in enumerate(splited_data):
            for c in self.C:
                for rho_0 in self.RHO_0:
                    weight = self.train(train, rho_0, c, epoch=epoch)
                    precision = self.test(test, weight)
                    print 'done fold %i' % idx, ' on [rho_0: %s, c: %s]' \
                          % (rho_0, c)
                    hyper_test[(rho_0, c)] += precision
        return map(lambda (x, y): (x, y/fold), hyper_test.iteritems())

    def train(self, data, rho_0, c, epoch=EPOCH):
        FEATURE_NUM = len(data[0][1])
        weight = [0 for _ in range(FEATURE_NUM)]
        t = 0
        for _ in range(epoch):
            shuffle(data)
            for y_i, x_i in data:
                r_t = self.get_learning_rate(rho_0, t, c)
                weight = self.update_weight(weight, x_i, y_i, r_t, c)
                t += 1
        return weight

    def get_learning_rate(self, rho_0, t, c):
        return rho_0/(1+float(rho_0*t)/c)

    def update_weight(self, weight, x_i, y_i, r_t, c):
        base = scale_vector(weight, 1-r_t)
        penalty = [0 for _ in base]
        if y_i*(dot_prod(weight, x_i)) <= 1:
            penalty = scale_vector(x_i, r_t*c*y_i)
        return verctor_add(penalty, base)


def compute_distance(datas):
    for data in datas:
        yield get_furthest(data)


def get_astro_data(TYPE):
    for cat in [ORIGIN, SCALED]:
        yield list(prepare_data(cat+TYPE))
        yield list(tranpose_data(cat+TYPE))


def main():
    label = ['original', 'original.trans', 'scaled', 'scaled.trans', 'data0']

    astro_train = list(get_astro_data('train'))
    astro_test = list(get_astro_data('test'))

    diss = list(compute_distance(astro_train))
    print '\nfurthest distance from origin', label[:-1]
    print diss

    train_data = astro_train + [list(prepare_data(FILE_TRAIN))]
    test_data = astro_test + [list(prepare_data(FILE_TEST))]

    for i in range(5):
        print '\n--------------------\n'+label[i]
        svm_clf = SVM(train_data[i], C, RHO_0)
        hyper_list = sorted(svm_clf.cross_validation(FOLD, EPOCH_CV),
                            key=lambda x: x[1], reverse=True)
        print '\ncross validation:\nrho_0, c, precision:'
        for (rho_0, c), precision in hyper_list[:5]:
            print rho_0, c, precision
        (rho_0, c), precision = hyper_list[0]
        print 'best rho_0, c:'
        print rho_0, c
        weight = svm_clf.train(train_data[i], rho_0, c)
        precision = svm_clf.test(test_data[i], weight)
        print '\nprecision: ', precision
        print 'margin(positive, negative): ', svm_clf.margin(test_data[i], weight)


if __name__ == "__main__":
    main()
