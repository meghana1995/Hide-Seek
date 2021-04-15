'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################
# python packages
import numpy as np
import math

# local files
import board_generator
import visible_squares


# Parameters for Grid
################################################################################
radius = 50
square_size = 7
hider_start = (30,30)
seeker_start = (0,0)
# seed = 1
seed = math.floor(np.random.rand()*456132545)
simplex_cutoffs = ( .67 , .67 )
simplex_scales = ( 10 , 5 )
opening_width = 3
second_pass = True


# Primary Export/Functionality
################################################################################

class Environment:
  '''
  Environment class used to represent the environment of our Hide & Seek AI
  problem. This environment will contain most of the data and methods our agents
  need to leverage to perceive the environment as the Hide & Seek game carries
  out.
  '''

  def __init__(self):
    '''
    Initializes new Environment instance.
    '''
    # board indicating walls in the environment
    self.board = board_generator.generateBoard(
      radius, seed, simplex_cutoffs, simplex_scales, opening_width, second_pass
    )

  def getMiddlePos(self):
    '''
    Returns the index of the position at the center of the board.
    '''
    ( n , m ) = self.board.shape
    return ( math.floor(n) , math.floor(m) )

  def isWall(self, position):
    '''
    Returns boolean value indicating if the given index position is a wall
    on the board.
    '''
    return self.board[position[0]][position[1]]

  def isOnBoard(self, position):
    '''
    Returns boolean value indicating if the given index position is on the
    board. I.E. is it a valid index within the dimensions of the board.
    '''
    ( n , m ) = self.board.shape
    return 0<=position[0]<n and 0<=position[1]<m

  def isValidPos(self, position):
    '''
    Returns boolean value indicating if the given index position is on the
    board and not a wall. I.E. is this a valid position for one of our agents
    to move to.
    '''
    return (not self.isWall(position)) and self.isOnBoard(position)

  def visibleNeighbors(self, position, distance):
    '''
    Returns a set containing all the positons within the given manhattan distance
    that are visible from the current position. In this context, visible means
    that a straight line can be drawn between the two positions without being
    obstructed by a wall on our board.
    '''
    return visible_squares.visibleTiles(position, distance, self.board)


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create environment
  test_environment = Environment()

  # print the board
  print(test_environment.board)