#!/usr/bin/python3

# coding: utf-8

import math
import sys, os
import tempfile
import argparse
import argh

import functools
import operator
import argparse
import numpy as np

from cchess import *
import tensorflow as tf
import util
from policy import PolicyNetwork
from CChessDataSet import CChessDataSet

def train_piece_net(save_file, restore_file=None, epochs=10, logdir=None, checkpoint_freq=10000):
    TRAIN_DATA_DIR = os.path.join(os.getcwd(), "tfr")
    CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoint")

    tfrecords = [os.path.join(TRAIN_DATA_DIR, 'training_piece.tfrecord')]
    train_dataset = CChessDataSet(tfrecords, batch_size=32)

    n = PolicyNetwork()
    try:
        n.initialize_variables(restore_file)
    except:
        n.initialize_variables(None)
    if logdir is not None:
        n.initialize_logging(logdir)

    n.train(train_dataset, save_file)


def train_move_net(piece_index, save_file, restore_file=None, epochs=10, logdir=None, checkpoint_freq=10000):
    TRAIN_DATA_DIR = os.path.join(os.getcwd(), "tfr")
    CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoint")

    tfrecords = [os.path.join(TRAIN_DATA_DIR, 'training_move%d.tfrecord' % piece_index)]
    train_dataset = CChessDataSet(tfrecords, batch_size=32)

    n = PolicyNetwork()
    try:
        n.initialize_variables(restore_file)
    except:
        n.initialize_variables(None)
    if logdir is not None:
        n.initialize_logging(logdir)

    n.train(train_dataset, save_file)


parser = argparse.ArgumentParser()
parser.add_argument('--nettype', type=str, default='piece',
                    help='network type: piece | move')
parser.add_argument('--piecename', type=str, default='P',
                    help='piece name: P | R | N | B | A | K | C')
FLAGS = parser.parse_args()

# if FLAGS.nettype == 'piece':
#     save_file = os.path.join(os.path.join(os.getcwd(), "checkpoint_shuffle_epoch"), 'model.ckpt')
#     train_piece_net(save_file)
# else:
#     try:
#         piece_index = util.PIECE_TO_INDEX[FLAGS.piecename]
#         save_file = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'move%d-model.ckpt' % piece_index)
#         train_move_net(piece_index, save_file)
#     except KeyError:
#         parser.print_help()

for i in range(0, len(util.PIECE_TO_INDEX)):
    piece_index = i
    print('training piece %s of movenet...' % util.INDEX_TO_PIECE[piece_index])
    save_file = os.path.join(os.path.join(os.getcwd(), "checkpoint"), 'move%d-model.ckpt' % piece_index)
    train_move_net(piece_index, save_file)
