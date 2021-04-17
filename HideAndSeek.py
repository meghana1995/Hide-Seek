'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################

# python packages
from Visualization import Visualization
import numpy as np
import math

# local files
from Environment import Environment
from HidingAgent import HidingAgent
from SeekingAgent import SeekingAgent
# from Visualization import Visualization


# Constants
################################################################################


# Primary Export/Functionality
################################################################################

class HideAndSeek:
  '''
  Class for representing a game of Hide & Seek. This class will contain the
  environment and agents necessary for carrying out a game of Hide & Seek.
  Within this class are the various methods for each of the parts of the game.
  '''

  def __init__(self, hiding_time, vision_range):
    '''
    Initializes new Hiding Agent instance.
    '''
    # set the environment and agents for the game
    self.environment = Environment()
    self.start_position = self.environment.getMiddlePos()
    self.env_shape = self.environment.board.shape
    self.hiding_agent = HidingAgent(self.env_shape, self.start_position, vision_range)
    self.seeking_agent = SeekingAgent(self.env_shape, self.start_position, vision_range)
    # set the amount of time allowed for the agent to hide
    self.hiding_time = hiding_time
    # initialize internal clock for the game
    self.clock = 0

  def tickClock(self):
    '''
    Increases internal clock and environment clock by a value of 1.
    '''
    self.clock += 1

  def resetClock(self):
    '''
    Resets the internal clock to 0.
    '''
    self.clock = 0

  def resetAgents(self):
    '''
    Resets belief states of the agents.
    '''
    self.hiding_agent.resetState(self.start_position)
    self.seeking_agent.resetState(self.start_position)

  def resetEnv(self):
    '''
    Creates a new environment for the game and resets agents for said
    environment.
    '''
    self.environment = Environment()
    self.start_position = self.environment.getMiddlePos()
    self.env_shape = self.environment.board.shape
    self.resetAgents()

  def foundHider(self,open_set,wall_set,hider_position):
    '''
    Determines if the seeker found the hider. In other words, this function
    checks if the hider is positioned in one of the perceivable squares. If so,
    it returns the hider position, else it returns None.
    '''
    found_hider = hider_position in open_set or hider_position in wall_set
    if found_hider:
      return hider_position
    else:
      return None

  def hidingStep(self):
    '''
    This function iterates through a single step of the game for the Hiding
    sequence. That is, it allows the Hiding agent to decide and perform an
    action and updates the game accordingly.
    '''
    # let agent perceive the environment and update belief state
    open , walls = self.environment.perceiveEnv(self.hiding_agent)
    self.hiding_agent.updateState(open, walls, self.clock)
    # get next action from agent and allow it to perform said action
    action = self.hiding_agent.getAction()
    self.hiding_agent.performAction(action)
    # update game clock
    self.tickClock()

  def seekingStep(self):
    '''
    This function iterates through a single step of the game for the Hiding
    sequence. That is, it allows the Hiding agent to decide and perform an
    action and updates the game accordingly.
    '''
    # let agent perceive the environment and update belief state
    open , walls = self.environment.perceiveEnv(self.seeking_agent)
    hider_position = self.foundHider(open,walls,self.hiding_agent.position)
    self.seeking_agent.updateState(open,walls,hider_position)
    # get next action from agent and allow it to perform said action
    action = self.seeking_agent.getAction()
    self.seeking_agent.performAction(action)
    # update game clock
    self.tickClock()

  def simulateGame(self):
    '''
    This function simulates an entire game of Hide & Seek.
    '''
    # print start message
    print("Starting Hide & Seek Simulation")

    # hiding segment of game
    for i in range(self.hiding_time):
      self.hidingStep()
      print("Step {} Hiding Agent Pos: {}".format(self.clock,self.hiding_agent.position))

    # reset clock
    self.resetClock()
    
    # seeking segment of game
    while (self.seeking_agent.position != self.hiding_agent.position):
      self.seekingStep()
      print("Step {} Seeking Agent Pos: {}".format(self.clock,self.seeking_agent.position))

    # print end message
    print("Finished Hide & Seek Simulation")
    print("Seeker Time: {}".format(self.clock))

  def visualizeGame(self):
    '''
    This function simulates and visualizes an entire game of Hide & Seek.
    '''
    # print start message
    print("Starting Hide & Seek Simulation")

    # initialize visualization
    visualization = Visualization(self.environment, self.hiding_agent, self.seeking_agent)

    # hiding segment of game
    for i in range(self.hiding_time):
      self.hidingStep()
      visualization.moveHider()
      print("Step {} Hiding Agent Pos: {}".format(self.clock,self.hiding_agent.position))

    # reset clock
    self.resetClock()
    
    # seeking segment of game
    while (self.seeking_agent.position != self.hiding_agent.position):
      self.seekingStep()
      visualization.moveSeeker()
      print("Step {} Seeking Agent Pos: {}".format(self.clock,self.seeking_agent.position))

    # print end message
    print("Finished Hide & Seek Simulation")
    print("Seeker Time: {}".format(self.clock))

    # end visualization
    visualization.endVis()


# Unit Tests
################################################################################
if __name__ == "__main__":
  
  # # test simulation
  # game = HideAndSeek(300,4)
  # game.simulateGame()

  # test simulation
  game = HideAndSeek(50,3)
  game.visualizeGame()
  
    
