'''
Created on Apr 9, 2021

@author: young
'''

# Imports
################################################################################



# Functions to Export
################################################################################
def ManhattanDist(pos1,pos2):
  '''
  Manhattan distance between two points.
  '''
  return abs( pos1[0] - pos2[0] ) + abs( pos1[1] - pos2[1] )

def normalize(values):
  '''
  Normalizes given values. That is, divide every value by the sum of the values
  so that the results can be interpreted as probablities. Assumes each value
  is non-negative.
  '''
  total = sum(values)
  if (total == 0):
    n = len(values)
    return [1 / n for value in values]
  else:
    return [value / total for value in values]


# Helper Functions
################################################################################
