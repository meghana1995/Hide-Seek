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

class HidingAgent(Agent):
  '''
  Class for representing our Hiding Agent in the game of Hide & Seek.
  '''

  def __init__(self, algorithm, env_shape, start_pos, vision_range, h_weights, hiding_time):
    '''
    Initializes new Hiding Agent instance.
    '''
    # call to super
    super().__init__(algorithm, env_shape, start_pos, vision_range, h_weights)
    # store hiding_time and time left on clock for finding hiding place
    self.hiding_time = hiding_time
    self.time_left = hiding_time
    # store hiding position found so far
    self.best_hiding_place = start_pos
    self.best_score = 0
    self.current_score = 0 # for testing
    # store path back to this best position
    self.path_to_best = []
    # store number of actions taken since finding hiding place (for knowing when to calculate path back)
    self.moves_since_best = 0
    # boolean that indicates if agent is on its way back to its hiding place
    self.hiding = False

  def resetState(self):
    '''
    Resets the agent's belief state back to its original state.
    '''
    super().resetState()
    self.time_left = self.hiding_time
    self.best_hiding_place = self.start_position
    self.best_score = 0
    self.path_to_best = []
    self.moves_since_best = 0
    self.hiding = False


  def updateState(self, open_squares, wall_squares, visibilityTable, game_clock):
    '''
    This function allows the agent to update its belief state of the environment
    based off the percepts it receives.
    '''
    # call super to update environment based off visible squares
    super().updateState(open_squares, wall_squares, visibilityTable)
    # update game clock
    self.time_left = self.hiding_time - game_clock
    # check if current position is better hiding place
    hider_score = self.hideability(self.position)
    self.current_score = hider_score
    if (hider_score >= self.best_score):
      self.best_score = hider_score
      self.best_hiding_place = self.position
      self.path_to_best = []
      self.moves_since_best = 0
    # if agent is currently following path to hiding place no other changes
    if (self.hiding):
      pass
    # else run checks for when time is running out, to ensure we can hide
    else:
      # recalculate path to best position if time running out
      if (len(self.path_to_best)+self.moves_since_best+1 > self.time_left-1):
        print("Recalculating Path To Hiding Place")
        self.path_to_best = self.aStar(self.best_hiding_place)
        print(self.path_to_best)
        self.moves_since_best = 0
      # check if we need to start following path back to hiding place
      if (len(self.path_to_best)+1 > self.time_left-1):
        self.hiding = True    
    
    print("Time Left: {}".format(self.time_left))
    print("Path Length: {}".format(len(self.path_to_best)))
    print("Moves Made: {}".format(self.moves_since_best))

  def performAction(self, action):
    '''
    Makes the agent perform the given action and updates agent's internal
    data accordingly.
    '''
    # call to super to update agent's position and path from start
    super().performAction(action)
    # update number of moves taken since last calculating path to hiding place
    self.moves_since_best += 1


  def getAction(self):
    '''
    This funciton represents the Hiding Agent determining what action to carry
    out next while trying to find its hiding place.
    '''
    # if time is running out, follow path back to best hiding place
    if (self.hiding):
      if (self.position == self.best_hiding_place):
        return "nothing"
      else: 
        return self.path_to_best.pop(0)
    # else pick appropriate action using agent's choice of algorithm
    else:
      if (self.algorithm == "dfs"):
        return self.dFS()
      elif (self.algorithm == "hc"):
        return self.hC()
      elif (self.algorithm == "ihc"):
        return self.improvedHC()
      elif (self.algorithm == "rhc"):
        return self.randomHC()
      else:
        return self.randomAction()


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test agent
  test_agent = HidingAgent((11,11), (5,5), 3)

  # print environment
  print(test_agent.environment)
    
