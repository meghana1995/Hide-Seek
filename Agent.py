'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################

# python packages
import heapq
import math
import numpy as np

# local files
from Environment import Environment
import helper_util


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
    # initialize agent's plan as None
    self.plan = None

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

  def validAction(self,position,action):
    '''
    Checks if given action is valid. Action is said to be valid if the action
    moves the agent into a square that is open.
    '''
    change = ALLOWED_ACTIONS[action]
    new_position = ( 
      position[0] + change[0],
      position[1] + change[1],
    )
    return self.environment[new_position] == 0

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
    self.plan = None

  def randomAction(self):
    '''
    Returns a random valid action
    '''
    valid_actions = self.validActions()
    random_index = math.floor(np.random.random()*len(valid_actions))
    return valid_actions[random_index]

  def aStar(self,goal):
    '''
    Returns list of actions that take the agent from it's current
    position to the given goal position.
    '''
    # initialize priority queue using current position and set of visited positions
    count = 0
    queue = [(
      0,
      count,
      { 
        "g": 0,
        "h": 0,
        "position": self.position,
        "path": [],
      }
    )]
    visited = set()
    # iteratively extend paths by pulling off queue and extending position
    while (len(queue) > 0):
      # pop off shortest path and add to vistited
      current = heapq.heappop(queue)[2]
      visited.add(current["position"])
      # if we reached the goal, then return its path
      if (current["position"] == goal):
        return current["path"]
      # extend path to each of this positions neighbors
      for action, change in ALLOWED_ACTIONS.items():
        # get new position for action
        new_position = ( 
          current["position"][0] + change[0],
          current["position"][1] + change[1],
        )
        # add position to queue if valid and not already visited
        if (new_position not in visited and self.environment[new_position] == 0):
          # calculate g and h
          g = current["g"] + 1
          h = helper_util.ManhattanDist(new_position,goal)
          # extend path
          new_path = current["path"] + [action]
          # define next object to add to queue
          count += 1
          next = { 
            "g": g,
            "h": h,
            "position": new_position,
            "path": new_path,
          }
          heapq.heappush(queue, ( g+h, count, next ))
    # return None if we reach here (no path found)
    return None

      


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test agent
  test_agent = Agent((11,11), (5,5), 3)

  # print environment
  print(test_agent.environment)


    
