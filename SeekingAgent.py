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
from Agent import Agent


# Primary Export/Functionality
################################################################################

class SeekingAgent(Agent):
  '''
  Class for representing our Seeking Agent in the game of Hide & Seek.
  '''

  def __init__(self, env_shape, start_pos, vision_range):
    '''
    Initializes new Seeking Agent instance.
    '''
    # call to super
    super().__init__(env_shape, start_pos, vision_range)

  def getAction(self):
    '''
    This funciton represents the Seeking Agent determining what action to carry
    out next while trying to find the Hiding Agent.
    '''
    # for now just returns a random valid acton
    return self.randomAction()


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test agent
  test_agent = SeekingAgent((11,11), (5,5), 3)

  # print environment
  print(test_agent.environment)
    
