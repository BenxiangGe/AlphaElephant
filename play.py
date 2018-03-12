#!/usr/bin/python3

# coding: utf-8

import sys, os
import argparse
import numpy as np

from cchess import *
import tensorflow as tf
import util

from policy import PolicyNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--fen', type=str, default='rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1',
                    help='fen of board')
FLAGS = parser.parse_args()


def eval_piece(network, board):
    model_file = os.path.join(os.path.join(os.getcwd(), 'checkpoint'), 'piece-model.ckpt')
    network.initialize_variables(model_file)
    im = util.convert_bitboard_to_image(board)
    piece_probs = network.run(im.reshape(-1, util.Y_SIZE, util.X_SIZE, util.PIECE_SIZE))
    # print("piece_probs: ", piece_probs)
    pred_piece = np.argmax(piece_probs)
    coordinate = util.score_to_coordinate(pred_piece)

    piece_index = np.argmax(im[coordinate[0], coordinate[1]])

    print("\n\nselected piece: ", util.INDEX_TO_PIECE[piece_index])
    print("selected move: from ", coordinate)

    # n.close()
    return piece_index


def eval_move(network, board, piece_index):
    model_file = os.path.join(os.path.join(os.getcwd(), 'checkpoint'), 'move%d-model.ckpt' % piece_index)
    network.initialize_variables(model_file)
    im = util.convert_bitboard_to_image(board)
    move_probs = network.run(im.reshape(-1, util.Y_SIZE, util.X_SIZE, util.PIECE_SIZE))
    # print("move_probs: ", move_probs)
    pred_move = np.argmax(move_probs)

    coordinate = util.score_to_coordinate(pred_move)
    print("selected move: to ", coordinate)

    return coordinate

board = ChessBoard(FLAGS.fen)

n = PolicyNetwork(use_cpu=True)
piece_index = eval_piece(n, board)
move_coordinate = eval_move(n, board, piece_index)

n.close()
