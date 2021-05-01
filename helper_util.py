'''
Created on Apr 9, 2021

@author: young
'''

# Imports
################################################################################
import numpy as np



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

def weightedNormalize(values,r):
  '''
  Normalizes given values but does so with a sense of randomness. Basically, as
  r goes to 1 the value distribution becomes completely random and uniform. But
  as r goes to 0, the distribution becomes more deterministic, and the best
  values get closer to 1 while the others go to 0. For this to work, r should
  be chosen from the range [0,1].
  '''
  p = np.array(values)
  m = p.max()
  if (m == 0):
    p = np.full(p.shape,1/p.shape[0])
  elif (r == 0):
    p = np.where(p < m, 0, 1)
  else:
    w = 1/r - 1
    p = np.exp(w*(p/m-1))
  # normalize result
  p = p / p.sum()
  return p


# Helper Functions
################################################################################
