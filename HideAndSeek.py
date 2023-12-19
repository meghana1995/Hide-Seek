'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################

# python packages
import numpy as np
import math
import time

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

  def __init__(self, environment, hiding_agent, seeking_agent, hiding_time, sleep_time):
    '''
    Initializes new Hiding Agent instance.
    '''
    # save environment
    self.environment = environment
    # save agents
    self.hiding_agent = hiding_agent
    self.seeking_agent = seeking_agent
    # save other parameters
    self.hiding_time = hiding_time
    self.sleep_time = sleep_time
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
    self.hiding_agent.resetState()
    self.seeking_agent.resetState()
    self.resetClock()

  def resetEnv(self):
    '''
    Creates a new environment for the game and resets agents for said
    environment.
    '''
    self.environment.resetEnv()
    self.resetAgents()

  def foundHider(self,open_set,wall_set,hider_position):
    '''
    Determines if the seeker found the hider. In other words, this function
    checks if the hider is positioned in one of the perceivable squares. If so,
    it returns the hider position, else it returns None.
    '''
    if found_hider := hider_position in open_set or hider_position in wall_set:
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
    open , walls , visibilityTable = self.environment.perceiveEnv(self.hiding_agent)
    self.hiding_agent.updateState(open, walls, visibilityTable, self.clock)
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
    open , walls , visibilityTable = self.environment.perceiveEnv(self.seeking_agent)
    hider_position = self.foundHider(open,walls,self.hiding_agent.position)
    self.seeking_agent.updateState(open,walls,visibilityTable,hider_position)
    # get next action from agent and allow it to perform said action
    action = self.seeking_agent.getAction()
    self.seeking_agent.performAction(action)
    # update game clock
    self.tickClock()

  def simulateGame(self, print_updates=False):
    '''
    This function simulates an entire game of Hide & Seek.
    '''
    # print start message
    if (print_updates):
      print("Starting Hide & Seek Simulation")

    # hiding segment of game
    for i in range(self.hiding_time):
      self.hidingStep()
      if (print_updates):
        print("Step {} Hiding Agent Pos: {}".format(self.clock,self.hiding_agent.position))

    # reset clock
    self.resetClock()
    
    # seeking segment of game
    while (self.seeking_agent.position != self.hiding_agent.position):
      self.seekingStep()
      if (print_updates):
        print("Step {} Seeking Agent Pos: {}".format(self.clock,self.seeking_agent.position))

    # print end message
      if (print_updates):
        print("Finished Hide & Seek Simulation")
        print("Seeker Time: {}".format(self.clock))

    # return total elapsed time
    return self.clock

  def visualizeGame(self, print_updates=False):
    '''
    This function simulates and visualizes an entire game of Hide & Seek.
    '''
    # print start message
    if (print_updates):
      print("Starting Hide & Seek Simulation")

    # initialize visualization
    visualization = Visualization(self.environment, self.hiding_agent, self.seeking_agent, self.sleep_time)

    # hiding segment of game
    for i in range(self.hiding_time):
      self.hidingStep()
      visualization.moveHider()
      if (print_updates):
        print(
          "Step {} Hiding Agent Pos: {}  ,  Current Score = {:.3f}  ,  Best Score = {:.3f}"
          .format(self.clock,self.hiding_agent.position,self.hiding_agent.current_score,self.hiding_agent.best_score)
        )

    # reset clock
    self.resetClock()
    
    # seeking segment of game
    while (self.seeking_agent.position != self.hiding_agent.position):
      self.seekingStep()
      visualization.moveSeeker()
      if (print_updates):
        print("Step {} Seeking Agent Pos: {}".format(self.clock,self.seeking_agent.position))

    # print end message
    if (print_updates):
      print("Finished Hide & Seek Simulation")
      print("Seeker Time: {}".format(self.clock))

    # end visualization
    visualization.endVis()

    # return total elapsed time
    return self.clock


# Unit Tests
################################################################################
if __name__ == "__main__":

  ##########################
  # SIMULATION PARAMETERS
  ##########################
  hiding_time = 400
  vision_range = 3
  print_updates = False
  ##########################
  # VISUALIZATION PARAMETERS
  ##########################
  sleep_time = 0
  ##########################
  # AGENT PARAMETERS
  ##########################
  # hiding agent
  hiding_alg = "whc"
  hider_weights = (1/3,0,2/3)
  h_randomness = .5
  # seeking agent
  seeking_alg = "whc"
  seeker_weights = (1/2,0,1/2)
  s_randomness = .5
  ##########################


  # Environment
  environment = Environment(vision_range)
  env_shape = environment.board.shape
  middle_pos = environment.getMiddlePos()

  # Agents
  hiding_agent = HidingAgent(hiding_alg,env_shape,middle_pos,vision_range,hider_weights,h_randomness,hiding_time)
  seeking_agent = SeekingAgent(seeking_alg,env_shape,middle_pos,vision_range,seeker_weights,s_randomness)
  
  # create Hide & Seek Game
  game = HideAndSeek(
    environment,hiding_agent,seeking_agent,
    hiding_time,sleep_time
  )

  
  # # simulate game
  # game.simulateGame(print_updates)

  # visualize game
  game.visualizeGame(print_updates)

  # # time 1000 simulations using the same board
  # N = 10
  # M = 10
  # n = M*N
  # total_time = 0
  # print("Timing n={} Iterations of Hide & Seek Game".format(n))
  # print("Using M={} Different Environments".format(M))
  # print("With N={} Simulations per Environment".format(N))
  # t0 = time.time()
  # for i in range(M):
  #   for j in range(N):
  #     total_time += game.simulateGame(print_updates)
  #     game.resetAgents()
  #   game.resetEnv()
  # t1 = time.time()
  # print("Elapsed Time: {}".format(t1-t0))
  # print("Average Sim. Time: {}".format((t1-t0)/n))
  # print("Average Seeker Time: {}".format(total_time/n))

  
    
