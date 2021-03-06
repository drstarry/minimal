# coding: utf-8
import os

# File
CUR_FOLDER = os.path.dirname(os.path.realpath(__file__))
FILE_FOLDER = '/data%i/'
FILE_TRAIN = 'train%i.%i'
FILE_TEST = 'test%i.%i'
FILE_NUM = range(2)
FEATURE_NUM = range(10, 110, 10)
IMG_FOLDER = '/img_/'

# hyper parameters list
MU = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
R = [0.01, 0.05, 0.1, 0.5, 1.0]
EPOCH = 10
