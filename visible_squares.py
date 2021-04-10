'''
Created on Apr 9, 2021

@author: young
'''

import numpy as np
import time



# Primary Export/Functionality
################################################################################
def visibleTiles(position,distance,board):
  '''
  Returns a set containing all the positons within the given manhattan distance
  that are visible from the current position. In this context, visible means
  that a straight line can be drawn between the two positions without being
  obstructed by a wall on our board.
  '''
  visible_tiles = set()
  for loc in grabFrontier(position,distance):
    rayTrace(position,loc,board,visible_tiles)
  return visible_tiles

# Helper Functions
################################################################################
def rayTrace(start,end,board,tile_set):
  '''
  Traces a line between the two given positions and walks down this line to
  determine all the "visible" tiles it passes through.
  '''
  dx = abs(start[0] - end[0])
  dy = abs(start[1] - end[1])
  x = start[0]
  y = start[1]
  n = dx + dy
  x_inc = 1 if (end[0] > start[0]) else -1
  y_inc = 1 if (end[1] > start[1]) else -1
  error = dx - dy
  dx *= 2
  dy *= 2

  i = 0
  while (i < n and onBoard(x,y,board)):
    if (error > 0):
      x += x_inc
      error -= dy
      i += 1
      if (board[x][y]):
        break
      else:
        tile_set.add((x,y))
    elif (error < 0):
      y += y_inc
      error += dx
      i += 1
      if (board[x][y]):
        break
      else:
        tile_set.add((x,y))
    else:
      if (board[x+x_inc][y] and board[x][y+y_inc]):
        break
      else:
        if (not board[x+x_inc][y]):
          tile_set.add((x+x_inc,y))
        if (not board[x][y+y_inc]):
          tile_set.add((x,y+y_inc))
        i += 2
        x += x_inc
        error += dx
        y += y_inc
        error -= dy
        if (board[x][y]):
          break
        else:
          tile_set.add((x,y))

def grabFrontier(point,d):
  '''
  Grabs the positions of all the tiles at the edge of the visible range. These
  will be used to generate our rays.
  '''
  x = point[0]
  y = point[1]
  frontier = []
  for i in range(0,d):
    frontier.append(( x+d-i , y+i ))
    frontier.append(( x-i , y+d-i ))
    frontier.append(( x-d+i , y-i ))
    frontier.append(( x+i , y-d+i ))
  return frontier

def onBoard(x,y,board):
  '''
  Returns a boolean value indicating if the given x,y coordinate is a valid
  index for our board.
  '''
  n = board.shape[0]
  return 0<=x<=n and 0<=y<=n


# Unit Tests
################################################################################
def timeTest(n,d):
  '''
  Unit test to make sure the visibleTiles function is efficient and runs
  in a reasonable amount of time for the problem sizes we intend to use.
  '''
  N = n**2
  board = np.zeros((2*(d+1)+1,2*(d+1)+1),np.int)
  center = (d+1,d+1)
  t0 = time.time()
  for i in range(N):
    visible_tiles = visibleTiles(center,d,board)
  t1 = time.time()
  total = t1-t0
  print("Number of Instances: {}".format(N))
  print("Instance Depth: {}".format(d))
  print("Total Runtime: {}".format(total))
  print("Runtime Per Instance: {}".format(total/N))
  

def testVisibleTiles():
  '''
  Unit test for the visibleTiles function to make sure it is working properly.
  '''
  board = np.array([
    [True,True,True,True,True,True,True,True,True],
    [True,True,True,True,False,True,True,True,True],
    [True,True,True,False,False,False,True,True,True],
    [True,True,False,False,True,True,False,True,True],
    [True,False,False,True,False,False,False,False,True],
    [True,True,False,False,False,False,False,True,True],
    [True,True,True,False,False,False,True,True,True],
    [True,True,True,True,False,True,True,True,True],
    [True,True,True,True,True,True,True,True,True],
  ],np.bool_)
  distance = 3
  position = (4,4)
  visible_tiles = visibleTiles(position,distance,board)
  print(visible_tiles)
  for (x,y) in visible_tiles:
    board[x][y] = True
  print(board.astype(int))

def testRayTrace():
  '''
  Unit test for the rayTrace function to make sure it is working properly.
  '''
  test_board = np.array([
    [False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False],
    [False,False,False,False,False,False,False],
  ],np.bool_)
  tile_set = set()
  start = (3,1)
  end = (6,6)
  rayTrace(start,end,test_board,tile_set)
  print(tile_set)
  for point in tile_set:
    test_board[point[0]][point[1]] = True
  print(test_board)

def testFrontier():
  '''
  Unit test for the grabFrontier function to make sure it is working properly.
  '''
  d = 5
  test_board = np.zeros((2*d+1,2*d+1),np.int)
  center = (d,d)
  frontier = grabFrontier(center,d)
  for (x,y) in frontier:
    test_board[x][y] = 1
  print(test_board)

# Main for Running Unit Tests
################################################################################
if __name__ == "__main__":
  timeTest(200,5)
  # testVisibleTiles()
  # testRayTrace()
  # testFrontier()
