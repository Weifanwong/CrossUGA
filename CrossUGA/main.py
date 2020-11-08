from __future__ import division
from __future__ import print_function

import time
import os
import heapq

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp


from optimizer import OptimizerAE
from input_data import load_data
from model import GCNModelAE
from preprocessing import *
from scipy.io import loadmat,mmwrite
from scipy import linalg
from scipy.io import loadmat
from algorithm import main



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate_dis', 0.001, 'Learning rate for discriminator.')
flags.DEFINE_float('learning_rate_gen', 0.001, 'Initial learning rate for generator.')
flags.DEFINE_integer('epoch', 200, 'Number of epochs to train.')
flags.DEFINE_integer('d_epoch', 1, 'Number of epochs to train.') #1
flags.DEFINE_integer('g_epoch', 10, 'Number of epochs to train.') #5
flags.DEFINE_integer('circle_epoch', 50, 'Number of epochs to train.') #100
flags.DEFINE_integer('g_hidden1', 512 , 'Number of units in hidden layer 1.')
flags.DEFINE_integer('g_hidden2', 512, 'Number of units in hidden layer 2.') #128
flags.DEFINE_integer('d_hidden1', 512, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('dropout1', 0.05, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('batch_num', 300, 'scale of real/fake_edges batch')
flags.DEFINE_integer('feature_num', 538, 'scale of real/fake_edges batch')
flags.DEFINE_float('alpha', 0.3, 'scale of real/fake_edges batch')
flags.DEFINE_float('beta', 0.3, 'scale of real/fake_edges batch')
flags.DEFINE_float('AX_alpha', 0., 'scale of real/fake_edges batch')



run = main()
run.runner()


