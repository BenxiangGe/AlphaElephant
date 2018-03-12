#!/usr/bin/python2

# coding: utf-8

import sys, os
import argparse
import numpy as np

from cchess import *
import tensorflow as tf
import util

TRAIN_DATA_DIR = os.path.join(os.getcwd(), "tfr")

features={'label': tf.FixedLenFeature([], tf.int64),
          'image': tf.FixedLenFeature([], tf.string)
}
piece_filename_queue = tf.train.string_input_producer([os.path.join(TRAIN_DATA_DIR, 'training_piece.tfrecord')])
piece_reader = tf.TFRecordReader()
_, serialized_piece = piece_reader.read(piece_filename_queue)

piece_features = tf.parse_single_example(serialized_piece, features=features)
label = tf.cast(piece_features['label'], tf.int32)
image = tf.decode_raw(piece_features['image'], tf.float64)
#image = tf.reshape(image, [util.PIECE_SIZE, util.Y_SIZE, util.X_SIZE])
#image = tf.reshape(image, [util.Y_SIZE, util.X_SIZE, util.PIECE_SIZE])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(4):
        im, lbl = sess.run([image, label])
        print ("lbl: ", lbl)
        print ("im.shape: ", im.shape)
        print ("im:\n", im.reshape(7, 10, 9))

    coord.request_stop()
    coord.join(threads)

# move0_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move0.tfrecord'))
# move1_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move1.tfrecord'))
# move2_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move2.tfrecord'))
# move3_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move3.tfrecord'))
# move4_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move4.tfrecord'))
# move5_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move5.tfrecord'))
# move6_reader = tf.python_io.TFRecordReader(os.path.join(TRAIN_DATA_DIR, 'training_move6.tfrecord'))

