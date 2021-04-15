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
from HidingAgent import HidingAgent
from SeekingAgent import SeekingAgent
from Visualization import Visualization


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

  def __init__(self, environment, hiding_agent, seeking_agent, hiding_time):
    '''
    Initializes new Hiding Agent instance.
    '''
    # set the environment and agents for the game
    self.environment = environment
    self.hiding_agent = hiding_agent
    self.seeking_agent = seeking_agent
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

  def hidingStep(self):
    '''
    This function iterates through a single step of the game for the Hiding
    sequence. That is, it allows the Hiding agent to decide and perform an
    action and updates the game accordingly.
    '''
    action = self.hiding_agent.getAction(self.clock)
    self.hiding_agent.performAction(action)
    self.tickClock()

  def seekingStep(self):
    '''
    This function iterates through a single step of the game for the Hiding
    sequence. That is, it allows the Hiding agent to decide and perform an
    action and updates the game accordingly.
    '''
    action = self.seeking_agent.getAction()
    self.seeking_agent.performAction(action)
    self.tickClock()

  def simulateGame():
    '''
    This function simulates an entire game of Hide & Seek.
    '''
    # TODO
    pass

  def visualizeGame():
    '''
    This function simulates and visualizes an entire game of Hide & Seek.
    '''
    # TODO
    pass


# Unit Tests
################################################################################
if __name__ == "__main__":
  # Eventually will have this section run an example that visualizes a game
  # of Hide & Seek.
  # TODO
  pass

  
    
