# coding: utf-8
import os

# File
CUR_FOLDER = os.path.dirname(os.path.realpath(__file__))
FILE_FOLDER = '/data/'
FILE_TRAIN = 'train0.10'
FILE_TEST = 'test0.10'
IMG_FOLDER = '/img_/'
ORIGIN = '/original/'
SCALED = '/scaled/'

# hyper parameters list
C = [1, 10, 100, 1000, 1500]
RHO_0 = [0.0001, 0.001, 0.01, 0.1, 1, 10]
EPOCH = 30
EPOCH_CV = 10
FOLD = 10
