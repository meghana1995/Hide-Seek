'''
Created on Apr 30, 2021

@author: young
'''

# Imports
################################################################################

# python packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import time
import progressbar

# local files
from HideAndSeek import HideAndSeek
from Environment import Environment
from HidingAgent import HidingAgent
from SeekingAgent import SeekingAgent


# Default Agent Parameters
################################################################################
# game parameters
hiding_time = 400
vision_range = 3
sleep_time = 0


# Trial Function - (Main Function Used to Test Agent Performance)
################################################################################
def gameTrial(m, n, bins, h_alg, s_alg, h_weights=(1/3,0,2/3), s_weights=(1/2,0,1/2), h_randomness=.5, s_randomness=.5):
  # print trial has started
  print()
  message = "Trial For {} Hider vs {} Seeker: m = {} , n = {}".format(h_alg,s_alg,m,n)
  print(message)
  print("#"*len(message))
  # build game
  game = buildGame(h_alg, s_alg, h_weights, s_weights, h_randomness, s_randomness)
  # run simulations
  times = runTrial(game,m,n)
  # create histogram
  title = "{} Hider vs {} Seeker:".format(h_alg,s_alg)
  stats = buildHistrogram(times,bins,title)
  # return stats
  return stats


# Helper Functions
################################################################################
def buildGame(h_alg, s_alg, h_weights, s_weights, h_randomness, s_randomness):
  # build environment
  environment = Environment(vision_range)
  env_shape = environment.board.shape
  middle_pos = environment.getMiddlePos()
  # build agents
  hiding_agent = HidingAgent(h_alg,env_shape,middle_pos,vision_range,h_weights,h_randomness,hiding_time)
  seeking_agent = SeekingAgent(s_alg,env_shape,middle_pos,vision_range,s_weights,s_randomness)
  # create Hide & Seek Game
  game = HideAndSeek(
    environment,hiding_agent,seeking_agent,
    hiding_time,sleep_time
  )
  return game

def runTrial(game,m,n):
  N = m*n
  times = np.zeros(N)
  bar = progressbar.ProgressBar(maxval=N, \
      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  count = 0
  for i in range(m):
    for j in range(n):
      times[i*n+j] = game.simulateGame()
      count += 1
      bar.update(count)
      game.resetAgents()
    game.resetEnv()
  bar.finish()
  return times

def buildHistrogram(x,b,title):
  # build new figure
  plt.figure()
  # build histogram
  n, bins, patches = plt.hist(x, b, density=True, facecolor='green', alpha=0.75)
  # stats
  N = x.shape[0]
  mu = x.mean()
  sigma = x.std()
  # get stats and print them
  time_mean = x.mean()
  time_std = x.std()
  time_stderr = time_std / math.sqrt(N)
  stats = ( time_mean , time_std , time_stderr , "random vs random")
  print("Trial Stats: N={} , mean={} , std={:.2f} , stderr={:.2f}".format(N,time_mean,time_std,time_stderr))
  # add a 'best fit' line
  y = norm.pdf(bins, mu, sigma)
  l = plt.plot(bins, y, 'r--', linewidth=1)
  plt.xlabel('Seeker Time')
  plt.ylabel('Probability')
  s_mu = '$\mu$'
  s_sig = '$\sigma$'
  plt.title("{}: N={} , {}={:.0f} , {}={:.0f}".format(title,N,s_mu,mu,s_sig,sigma))
  # plt.axis([40, 160, 0, 0.03])
  plt.grid(True)
  plt.show(block=False)
  plt.pause(5)
  # return stats
  return stats


# Main
################################################################################
if __name__ == "__main__":

  # Parameters for Agents
  ########################
  h_weights = (1/3,0,2/3)
  s_weights = (1/2,0,1/2)
  h_randomness = .5
  s_randomness = .5

  # Number of Samples per Trial
  #############################
  m, n, bins = 100 , 10 , 50


  # Random Seeker Trials
  #########################################################
  s_alg = "random"
  # random H vs random S
  h_alg =  "random"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg)
  # DFS H vs random S
  h_alg =  "dfs"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg)
  # Hill Climbing H vs random S
  h_alg =  "hc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights)
  # Improved Hill Climbing H vs random S
  h_alg =  "ihc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights)
  # Randomized Hill Climbing H vs random S
  h_alg =  "rhc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights)
  # Weighted Randomized Hill Climbing H vs random S
  h_alg =  "whc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, h_randomness=h_randomness)


  # DFS Seeker Trials
  #########################################################
  s_alg = "dfs"
  # random H vs random S
  h_alg =  "random"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg)
  # DFS H vs random S
  h_alg =  "dfs"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg)
  # Hill Climbing H vs random S
  h_alg =  "hc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights)
  # Improved Hill Climbing H vs random S
  h_alg =  "ihc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights)
  # Randomized Hill Climbing H vs random S
  h_alg =  "rhc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights)
  # Weighted Randomized Hill Climbing H vs random S
  h_alg =  "whc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, h_randomness=h_randomness)


  # Improved (Backtracking) Hill Climbing Seeker Trials
  #########################################################
  s_alg = "ihc"
  # random H vs random S
  h_alg =  "random"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, s_weights=s_weights)
  # DFS H vs random S
  h_alg =  "dfs"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, s_weights=s_weights)
  # Hill Climbing H vs random S
  h_alg =  "hc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights)
  # Improved Hill Climbing H vs random S
  h_alg =  "ihc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights)
  # Randomized Hill Climbing H vs random S
  h_alg =  "rhc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights)
  # Weighted Randomized Hill Climbing H vs random S
  h_alg =  "whc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, h_randomness=h_randomness, s_weights=s_weights)


  # Randomized Hill Climbing Seeker Trials
  #########################################################
  s_alg = "rhc"
  # random H vs random S
  h_alg =  "random"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, s_weights=s_weights)
  # DFS H vs random S
  h_alg =  "dfs"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, s_weights=s_weights)
  # Hill Climbing H vs random S
  h_alg =  "hc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights)
  # Improved Hill Climbing H vs random S
  h_alg =  "ihc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights)
  # Randomized Hill Climbing H vs random S
  h_alg =  "rhc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights)
  # Weighted Randomized Hill Climbing H vs random S
  h_alg =  "whc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, h_randomness=h_randomness, s_weights=s_weights)


  # Randomized Hill Climbing Seeker Trials
  #########################################################
  s_alg = "whc"
  # random H vs random S
  h_alg =  "random"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, s_weights=s_weights, s_randomness=s_randomness)
  # DFS H vs random S
  h_alg =  "dfs"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, s_weights=s_weights, s_randomness=s_randomness)
  # Hill Climbing H vs random S
  h_alg =  "hc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights, s_randomness=s_randomness)
  # Improved Hill Climbing H vs random S
  h_alg =  "ihc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights, s_randomness=s_randomness)
  # Randomized Hill Climbing H vs random S
  h_alg =  "rhc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, s_weights=s_weights, s_randomness=s_randomness)
  # Weighted Randomized Hill Climbing H vs random S
  h_alg =  "whc"
  rVr_stats = gameTrial(m, n, bins, h_alg, s_alg, h_weights=h_weights, h_randomness=h_randomness, s_weights=s_weights, s_randomness=s_randomness)



  # Ask for User to Type "end" to close plots
  #########################################################
  print()
  print('Please input "end" to cloes the plots and program.')
  while True:
    user_input = input()
    if user_input.lower() == "end":
      break