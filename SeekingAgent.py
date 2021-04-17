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
    # store position of hider (None until hider perceived)
    self.hider_position = None

  def updateState(self, open_squares, wall_squares, visibilityTable, hider_position):
    '''
    This function allows the agent to update its belief state of the environment
    based off the percepts it receives.
    '''
    # call super to update environment based off visible squares
    super().updateState(open_squares, wall_squares, visibilityTable)
    # update game clock
    self.hider_position = hider_position

  def resetState(self):
    '''
    Resets the agent's belief state back to its original state.
    '''
    super().resetState()
    self.hider_position = None

  def getAction(self):
    '''
    This funciton represents the Seeking Agent determining what action to carry
    out next while trying to find the Hiding Agent.
    '''
    # if agent has plan then follow this plan
    if (self.plan is not None):
      return self.plan.pop(0)
    # else if hider position is know build a plan and execute first action
    elif (self.hider_position is not None):
      self.plan = self.aStar(self.hider_position)
      return self.plan.pop(0)
    # else use algorithm to pick next action
    else:
      # for now just returns a random valid acton
      return self.randomAction()


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test agent
  test_agent = SeekingAgent((11,11), (5,5), 3)

  # print environment
  print(test_agent.environment)
    
