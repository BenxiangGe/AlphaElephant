from cchess import *
import util

xqf = 'test.xqf'
game = read_from_xqf(xqf);
board = game.dump_moves()[0][0].board
im = util.convert_bitboard_to_image(board)
print "image:\n", im

im = util.flip_image(im)
print "flipped image:\n", im
