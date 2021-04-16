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

  def __init__(self, env_shape, start_pos, vision_range):
    '''
    Initializes new Agent instance.
    '''
    # initialize the agent's belief state of the environment as unknown
    # -1 == unknown , 0 == open , 1 == wall
    self.environment = np.full(env_shape, -1, np.int)
    # initialize agent's start position and label it as open
    self.position = start_pos
    self.environment[start_pos[0]][start_pos[1]] = 0
    # initialize agent's vision range
    self.vision_range = vision_range

  def performAction(self, action):
    '''
    Makes the agent perform the given action and updates agent's internal
    data accordingly. Returns True if action succesfully completed else
    returns False.
    '''
    # calculate new position based off action
    change = ALLOWED_ACTIONS[action]
    new_position = ( 
      self.position[0] + change[0],
      self.position[1] + change[1],
    )
    # set agent position to new position if valid
    if (self.environment[new_position] == 0):
      self.position = new_position
      return True
    else:
      return False

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
      if (self.environment[new_position] == 0):
        valid_actions.append(action)
    # return valid actions
    return valid_actions

  def updateState(self, open_squares, wall_squares):
    '''
    This function allows the agent to update its belief state of the environment
    based off the open and wall squares it is able to perceive.
    '''
    # update open_squares in belief state
    for square in open_squares:
      self.environment[square] = 0
    for square in wall_squares:
      self.environment[square] = 1

  def resetState(self,start_position):
    '''
    Resets the agent's belief state back to its original state.
    '''
    self.environment.fill(-1)
    self.environment[start_position] = 0

  def randomAction(self):
    '''
    Returns a random valid action
    '''
    valid_actions = self.validActions()
    random_index = math.floor(np.random.random()*len(valid_actions))
    return valid_actions[random_index]


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test agent
  test_agent = Agent((11,11), (5,5), 3)

  # print environment
  print(test_agent.environment)


    
