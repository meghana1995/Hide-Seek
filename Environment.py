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


# Parameters for Board
################################################################################

# Good radius = 25 setup
########################
radius = 25
square_size = 7
#seed = 1
seed = math.floor(np.random.rand()*4568132545)
simplex_cutoffs = ( .67 , .67 )
simplex_scales = ( 7 , 3.5 )
opening_width = 3
second_pass = True


# # Good radius = 50 setup
##########################
# radius = 50
# square_size = 7
# #seed = 1
# seed = math.floor(np.random.rand()*4568132545)
# simplex_cutoffs = ( .67 , .67 )
# simplex_scales = ( 10 , 5 )
# opening_width = 3
# second_pass = True


# Primary Export/Functionality
################################################################################

class Environment:
  '''
  Environment class used to represent the environment of our Hide & Seek AI
  problem. This environment will contain most of the data and methods our agents
  need to leverage to perceive the environment as the Hide & Seek game carries
  out.
  '''

  def __init__(self,distance):
    '''
    Initializes new Environment instance.
    '''
    # store distance
    self.distance = distance
    # board indicating walls in the environment
    self.board = board_generator.generateBoard(
      radius, seed, simplex_cutoffs, simplex_scales, opening_width, second_pass
    )
    print(self.board.shape)
    # matrix of sets that represent the squares visible from each position
    self.visibility_table = visible_squares.visibilityTable(self.board,distance)

  def resetEnv(self):
    seed = math.floor(np.random.rand()*4568132545)
    # board indicating walls in the environment
    self.board = board_generator.generateBoard(
      radius, seed, simplex_cutoffs, simplex_scales, opening_width, second_pass
    )
    # matrix of sets that represent the squares visible from each position
    self.visibility_table = visible_squares.visibilityTable(self.board,self.distance)


  def getMiddlePos(self):
    '''
    Returns the index of the position at the center of the board.
    '''
    ( n , m ) = self.board.shape
    return ( math.floor(n/2) , math.floor(m/2) )

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

  def perceiveEnv(self, agent):
    '''
    Returns info that each agent perceives from the environment. As of now, this
    includes 3 objects. The first two are sets of the open positions and wall
    positions visible from the agents current position. The third object, is
    a table (dictionary) that contains the visible open squares for each open
    square that is visible. I.e. the agent can percieve both the tiles visible
    from its position, and which of these tiles are visible with respect to one
    another.
    '''
    # get squares visible from current position
    open_set , walls_set = self.visibility_table[agent.position]
    # determine which of these open squares are visible to one another
    visibilityTable = {}
    for position in open_set:
      open , walls = self.visibility_table[position]
      visible = open.intersection(open_set)
      visible.add(agent.position)
      visibilityTable[position] = visible
    # return objects
    return open_set , walls_set , visibilityTable


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create environment
  test_environment = Environment()

  # print the board
  print(test_environment.board)