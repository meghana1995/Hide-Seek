'''
Created on Apr 8, 2021

@author: young
'''

from opensimplex import OpenSimplex
import numpy as np

def generateBoard(radius, seed, cutoffs, scales, opening_width, second_pass=False):
    '''
    Generates board as an numpy.array of boolean values. True values indicate walls
    and False values indicate on open board. True values populated using the Simplex
    noise function for the given cutoff value, scale, and random seed. Does a second
    pass over the unfilled areas (for adding more granularity) if prescribed. Also
    forces the center of the grid to be more open, to hopefully prevent closed mazes
    using the opening_width value.
    '''
    width = radius*2 +1
    board = np.zeros((width, width), np.bool_)
    simplex = OpenSimplex(seed)
    # first pass generates large scale features of board
    for i in range(0,width):
        for j in range(0,width):
            dist = centerDist(i,j,radius)
            if (dist >= (radius)):
                board[i][j] = True
            else:
                x = (i-radius)
                y = (j-radius)
                # first pass generates large scale features of board
                simplex_value = (simplex.noise2d(x/scales[0], y/scales[0]) + 1 )*sigmoid(dist,opening_width) / 2
                board[i][j] = (simplex_value > cutoffs[0])
                # second pass generates small scale features of board
                if (second_pass and board[i][j] != True):
                    simplex_value = (simplex.noise2d(x/scales[1], y/scales[1]) + 1 )*sigmoid(dist,opening_width) / 2
                    board[i][j] = (simplex_value > cutoffs[1])
            
    return board


def centerDist(i,j,radius):
    '''
    Calculates the Manhattan distance from the center of the board.
    '''
    x = i-radius
    y = j-radius
    return abs(x) + abs(y)

def sigmoid(x,y):
    return 1 / (1 + np.exp(-(x-y)))



