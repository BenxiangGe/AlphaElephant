#!/usr/bin/python2

# coding: utf-8

import sys, os
import argparse
import numpy as np

from cchess import *
import tensorflow as tf
import util

# parser = argparse.ArgumentParser\
# 	(description='Convert Chinese Chess data into TensorRecord arrays of size 6*8*8 with labels (pieces/moves)')

# parser.add_argument('--xqfdir', type=str, default='', 
# 	help='The XQF data directory')
# parser.add_argument('--outdir', type=str, default='', 
# 	help='The output TFR data directory')

# args = parser.parse_args()

# XQF_DATA_DIR = args.datadir
# TRAIN_DATA_DIR = args.outdir
XQF_DATA_DIR = os.path.join(os.getcwd(), "xqf")
# XQF_DATA_DIR = os.path.join(os.getcwd(), "bad")
TRAIN_DATA_DIR = os.path.join(os.getcwd(), "tfr")

if not os.path.isdir(TRAIN_DATA_DIR):
    os.mkdir(TRAIN_DATA_DIR)

# X, y = [], []
# p1_X, p2_X, p3_X, p4_X, p5_X, p6_X, p7_X = [], [], [], [], [], [], []
# p1_y, p2_y, p3_y, p4_y, p5_y, p6_y, p7_y = [], [], [], [], [], [], []

piece_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_piece.tfrecord'))

move0_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move0.tfrecord'))
move1_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move1.tfrecord'))
move2_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move2.tfrecord'))
move3_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move3.tfrecord'))
move4_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move4.tfrecord'))
move5_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move5.tfrecord'))
move6_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATA_DIR, 'training_move6.tfrecord'))

data_dir = os.walk(XQF_DATA_DIR)
# for f in os.listdir(XQF_DATA_DIR):
for root, dirs, files in data_dir:
    for f in files:
        # print("root: ", root, ", f: ", f)
        if len(f) <= 4 or (not f.endswith('.xqf')):
            continue
        f = os.path.join(root, f)
        print("f: ", f)
        game = read_from_xqf(f)
        if len(game.dump_moves()) == 0: continue

        for move_index, move in enumerate(game.dump_moves()[0]):
            im = util.convert_bitboard_to_image(move.board)

            from_pos = move.p_from
            to_pos = move.p_to

            index_piece = np.where(im[from_pos.y, from_pos.x] == util.RED)
            if len(index_piece[0]) > 0:
                color = util.RED
            else:
                index_piece = np.where(im[from_pos.y, from_pos.x] == util.BLACK)
                if len(index_piece[0]) > 0:
                    color = util.BLACK
                else: # no piece in the position?
                    print("problematic game. f: ", f, ", move_index: ", move_index, ", move: ", move.to_chinese())
                    raise ValueError

            # flip the board if BLACK
            if color == util.BLACK:
                im = util.flip_image(im)
                im = util.flip_color(im)
                from_pos = util.flip_coord2d(move.p_from)
                to_pos = util.flip_coord2d(move.p_to)

            index_piece = index_piece[0][0]

            from_pos = util.flatten_coord2d(from_pos)
            to_pos = util.flatten_coord2d(to_pos)

            # to get into form (C, H, W)
#        im = np.rollaxis(im, 2, 0)
#        im = np.rollaxis(im, 2, 1)

            # Filling the piece network
            # X.append(im)
            # y.append(from_pos)
            piece_features = tf.train.Features(feature= {
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[from_pos])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im.tostring()]))
            })
            example = tf.train.Example(features=piece_features)
            piece_writer.write(example.SerializeToString())

            # Filling the p_X and p_y array
            # p_X = "p%d_X" % (index_piece + 1)
            # p_X = eval(p_X)
            # p_X.append(im)

            # p_y = "p%d_y" % (index_piece + 1)
            # p_y = eval(p_y)
            # p_y.append(to_pos)

            move_features = tf.train.Features(feature= {
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[to_pos])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im.tostring()]))
            })
            example = tf.train.Example(features=move_features)
            move_writer = "move%d_writer" % (index_piece)
            move_writer = eval(move_writer)
            move_writer.write(example.SerializeToString())

piece_writer.close()

move0_writer.close()
move1_writer.close()
move2_writer.close()
move3_writer.close()
move4_writer.close()
move5_writer.close()
move6_writer.close()
