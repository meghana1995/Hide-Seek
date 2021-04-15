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
from Environment import Environment


# Constants
################################################################################

# allowed actions
ALLOWED_ACTIONS = {
  "right":  (1,0),
  "left":   (-1,0),
  "up":     (0,1),
  "down":   (0,-1),
}


# Primary Export/Functionality
################################################################################

class Agent:
  '''
  Class for representing a generic Agent in the Hide & Seek game. This will
  be extended to create both our Hiding and Seeking Agents.
  '''

  def __init__(self, environment, start_pos, vision_range):
    '''
    Initializes new Hiding Agent instance.
    '''
    # set the environment, start positon, and visible range of the agent
    self.environment = environment
    self.position = start_pos
    self.vision_range = vision_range

  def performAction(self, action):
    '''
    Makes the agent perform the given action and updates agent's internal
    data accordingly.
    '''
    # calculate new position based off action
    change = ALLOWED_ACTIONS[action]
    new_position = ( 
      self.position[0] + change[0],
      self.position[1] + change[1],
    )
    # set agent position to new position if valid
    if (self.environment.isValidPos(new_position)):
      self.position = new_position

  def validActions(self):
    '''
    Checks each of the allowable actions and returns a list of those that
    the agent is able to perform based off its current position in the
    environment.
    '''
    # init list for storing valid actions
    valid_actions = []
    # iterate over actions and test if it results in valid position
    for action, change in ALLOWED_ACTIONS.items():
      new_position = ( 
        self.position[0] + change[0],
        self.position[1] + change[1],
      )
      if (self.environment.isValidPos(new_position)):
        valid_actions.append(action)
    # return valid actions
    return valid_actions

  def visiblePositions(self):
    '''
    Returns a set containing all the positons within the given manhattan distance
    that are visible from the current position. In this context, visible means
    that a straight line can be drawn between the two positions without being
    obstructed by a wall on our board.
    '''
    return self.environment.visibleTiles(self.position, self.vision_range)



# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test environment
  test_environment = Environment()

  # create test agent
  test_agent = Agent()

  
    
