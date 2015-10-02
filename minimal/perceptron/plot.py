# coding: utf-8

import matplotlib.pyplot as plt

from perceptron import prepare_data, train_batch
from const import FEATURE_NUM, IMG_FOLDER, CUR_FOLDER

path = CUR_FOLDER + IMG_FOLDER


def plot_dot(x, y, data='0', y_label='number of updates', margin=False,
             agg=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not margin:
        title = 'simple perceptron'
    else:
        title = 'margin perceptron'
    ax.set_title(title)
    ax.set_xlabel('dimension')
    ax.set_ylabel(y_label)
    ax.plot(x, y, 'o')
    x_lo, x_hi = min(x), max(x)
    y_lo, y_hi = min(y), max(y)
    ax.axis([x_lo-10, x_hi+10, y_lo-0.01, y_hi+0.01])
    if not agg:
        plt.savefig(path+data+title+y_label+' not agg.png')
    else:
        plt.savefig(path+data+title+y_label+' agg.png')


def precoss_data():
    data = prepare_data()

    # simple perceptron
    mistakes, precisions = train_batch(data, r=0.8)
    plot_dot(FEATURE_NUM, mistakes[0][0])
    plot_dot(FEATURE_NUM, precisions[0][0], y_label='precision')
    plot_dot(FEATURE_NUM, mistakes[1][0], data='1')
    plot_dot(FEATURE_NUM, precisions[1][0], data='1', y_label='precision')

    # margin perceptron
    mistakes, precisions = train_batch(data, mu=0.1, r=0.8)
    plot_dot(FEATURE_NUM, mistakes[0][1], margin=True)
    plot_dot(FEATURE_NUM, precisions[0][1], margin=True, y_label='precision')
    plot_dot(FEATURE_NUM, mistakes[1][1], margin=True,
             data='1')
    plot_dot(FEATURE_NUM, precisions[1][1], margin=True, y_label='precision',
             data='1')

    # simple aggressive perceptron
    mistakes, precisions = train_batch(data, r=0.8, agg=True)
    plot_dot(FEATURE_NUM, mistakes[0][0], agg=True)
    plot_dot(FEATURE_NUM, precisions[0][0], agg=True, y_label='precision')
    plot_dot(FEATURE_NUM, mistakes[1][0], agg=True, data='1')
    plot_dot(FEATURE_NUM, precisions[1][0], agg=True, data='1',
             y_label='precision')

    # margin perceptron
    mistakes, precisions = train_batch(data, mu=0.1, r=0.8, agg=True)
    plot_dot(FEATURE_NUM, mistakes[1][0], margin=True, agg=True)
    plot_dot(FEATURE_NUM, precisions[1][0], margin=True, y_label='precision',
             agg=True)
    plot_dot(FEATURE_NUM, mistakes[1][1], margin=True,
             agg=True, data='1')
    plot_dot(FEATURE_NUM, precisions[1][1], margin=True,
             y_label='precision', agg=True, data='1')

if __name__ == "__main__":
    precoss_data()
