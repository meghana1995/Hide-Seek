'''
Created on Apr 9, 2021

@author: young
'''

# Imports
################################################################################
import numpy as np
import time


# Primary Export/Functionality
################################################################################
def visibleTiles(position,distance,board):
  '''
  Returns two sets, one containing all the open positons within the given
  manhattan distance that are visible from the current position and the other
  containing the visible walls. In this context, visible means that a straight
  line can be drawn between the two positions without being obstructed by a
  wall on our board.
  '''
  open_squares = set()
  wall_squares = set()
  for loc in grabFrontier(position,distance):
    rayTrace(position,loc,board,open_squares,wall_squares)
  return open_squares, wall_squares


# Helper Functions
################################################################################
def rayTrace(start,end,board,open_set,wall_set):
  '''
  Traces a line between the two given positions and walks down this line to
  determine all the "visible" tiles it passes through.
  '''
  # initialize variables for walking down line
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

  # walk down line until we hit a wall, are off the board, or reach the end of the line
  # while doing so, keep track of all tiles we encounter as "visible" tiles
  i = 0
  while (i < n and onBoard(x,y,board) and not board[x,y]):
    # if error is > 0 then move horizontally
    if (error > 0):
      # move to next tile
      x += x_inc
      error -= dy
      i += 1
      # add next tile to appropriate set, break if it is a wall
      if (board[x,y]):
        wall_set.add((x,y))
        break
      else:
        open_set.add((x,y))
    # if error is < 0 then move vertically
    elif (error < 0):
      # move to next tile
      y += y_inc
      error += dx
      i += 1
      # add next tile to appropriate set, break if it is a wall
      if (board[x,y]):
        wall_set.add((x,y))
        break
      else:
        open_set.add((x,y))
    # else error is = 0 so move vertically and horizontally
    else:
      # if both tiles on corner are walls then stop
      if (board[x+x_inc,y] and board[x,y+y_inc]):
        wall_set.add((x+x_inc,y))
        wall_set.add((x,y+y_inc))
        break
      # else at least one tile is open so keep walking down line
      else:
        # check if each tile on corner
        if (board[x+x_inc,y]):
          wall_set.add((x+x_inc,y))
        else:
          open_set.add((x+x_inc,y))
        if (board[x,y+y_inc]):
          wall_set.add((x,y+y_inc))
        else:
          open_set.add((x,y+y_inc))
        # move to next tile
        i += 2
        x += x_inc
        error += dx
        y += y_inc
        error -= dy
        # add next tile to appropriate set, break if it is a wall
        if (board[x,y]):
          wall_set.add((x,y))
          break
        else:
          open_set.add((x,y))


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
  (n,m) = board.shape
  return 0<=x<n and 0<=y<m


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
    open , walls = visibleTiles(center,d,board)
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
  open , walls = visibleTiles(position,distance,board)
  output = np.zeros((9,9),np.int)
  print(open)
  print(walls)
  for (x,y) in open:
    output[x][y] = 1
  for (x,y) in walls:
    output[x][y] = 2
  print(output)

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
  open_set = set()
  wall_set = set()
  start = (3,1)
  end = (6,6)
  rayTrace(start,end,test_board,open_set,wall_set)
  print(open_set)
  print(wall_set)
  for point in open_set:
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
  timeTest(100,5)
  # testVisibleTiles()
  # testRayTrace()
  # testFrontier()
