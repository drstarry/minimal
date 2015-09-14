# coding: utf-8
import os

# File
CUR_FOLDER = os.path.dirname(os.path.realpath(__file__))
FILE_FOLDER = '/tic_tac_toe/'
FILE_TRAIN = 'tic-tac-toe-train-'
FILE_TEST = 'tic-tac-toe-test.txt'

# Data description
FEATURE_NUM = 9
VALUES = 'box'

# File number(k-fold crossvalidation)
K = 6
