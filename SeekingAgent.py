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

  def __init__(self, environment, start_pos, vision_range):
    '''
    Initializes new Seeking Agent instance.
    '''
    # call to super
    super().__init__(self, environment, start_pos, vision_range)

  def getAction(self):
    '''
    This funciton represents the Seeking Agent determining what action to carry
    out next while trying to find the Hiding Agent.
    '''
    # TODO
    pass
    
