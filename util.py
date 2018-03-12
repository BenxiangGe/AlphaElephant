
# coding: utf-8

import functools
import operator

import numpy as np

from cchess import *

RED = 1
BLACK = -1
EMPTY = 0

X_SIZE, Y_SIZE, PIECE_SIZE = 9, 10, 7
BOARD_SIZE = Y_SIZE * X_SIZE
IMAGE_SIZE = (Y_SIZE, X_SIZE, PIECE_SIZE)

PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'A' : 4, 'K' : 5, 'C' : 6}
INDEX_TO_PIECE = {0 : 'P', 1 : 'R', 2 : 'N', 3 : 'B', 4 : 'A', 5 : 'K', 6 : 'C'}


def convert_bitboard_to_image(board):
    im = np.zeros(IMAGE_SIZE)

    for y in range(Y_SIZE):
        for x in range(X_SIZE):
            fench = board.get_fench(Pos(x, y))
            if fench is None: continue

            if fench.isupper():
                im[y, x, PIECE_TO_INDEX[fench.upper()]] = RED
            else:
                im[y, x, PIECE_TO_INDEX[fench.upper()]] = BLACK
    return im

def flip_image(im):
#return im[:, ::-1, :]
    return im[::-1, :, :]

def flip_color(im):
	indices_red = np.where(im == 1)
	indices_black = np.where(im == -1)
	im[indices_red] = -1
	im[indices_black] = 1
	return im

def flip_coord2d(pos):
	return Pos(pos.x, Y_SIZE - 1 - pos.y)

def flatten_coord2d(pos):
	return X_SIZE * pos.y + pos.x

    #convenience functions for initializing weights and biases
def product(numbers):
    return functools.reduce(operator.mul, numbers)

def score_to_coordinate(score):
    return (score // X_SIZE, score % X_SIZE)
