'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################

# python packages
import tensorflow as tf
import numpy as np
import functools
import time

# local files
from Environment import Environment
from HidingAgent import HidingAgent
from HideAndSeek import HideAndSeek
from RLAgentFeat import *



# Main - (For Training and Saving Model)
################################################################################
if __name__ == "__main__":
  ### Hyperparameters and setup for training ###
  ##############################################
  # Parameters for Env. and Hiding Agent
  hiding_time = 100
  vision_range = 3
  hiding_alg = "rhc"
  hider_weights = (1/3,0,2/3)
  h_randomness = .5

  # Hyperparameters for Training
  initial_learning_rate = 1e-2
  decay_steps = 10
  decay_rate = .99
  MAX_ITERS = 300 # increase the maximum to train longer
  batch_size = 100 # number of batches to run

  # Create Model
  model = createModel()

  # Optimizer
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)
  # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  optimizer = tf.keras.optimizers.Adam(initial_learning_rate)
  iteration = 0 # counter for training steps

  # # Comment Out When Using Parallel Runs
  # ################################################################################
  # # Initialize Environment
  # environment = Environment(vision_range)
  # env_shape = environment.board.shape
  # middle_pos = environment.getMiddlePos()

  # # Initialize Agents (Including RL Model in Seeking Agent)
  # hiding_agent = HidingAgent(hiding_alg,env_shape,middle_pos,vision_range,hider_weights,h_randomness,hiding_time)
  # seeking_agent = RLAgent(env_shape,middle_pos,vision_range, model)

  # # Initialize Hide and Seek Game
  # game = HideAndSeek(environment,hiding_agent,seeking_agent,hiding_time)
  # ################################################################################

  # Initialize Parallel Games
  ################################################################################
  # Initialize Environment
  envs = [Environment(vision_range) for _ in range(batch_size)]
  env_shape = envs[0].board.shape
  middle_pos = envs[0].getMiddlePos()

  # Initialize Agents (Including RL Model in Seeking Agent)
  hiders = [
    HidingAgent(hiding_alg,env_shape,middle_pos,vision_range,hider_weights,h_randomness,hiding_time) 
    for _ in range(batch_size)
  ]
  seekers = [
    RLAgent(env_shape,middle_pos,vision_range, model) 
    for _ in range(batch_size)
  ]

  # Initialize Hide and Seek Game
  games = [HideAndSeek(envs[b],hiders[b],seekers[b],hiding_time) for b in range(batch_size)]
  ################################################################################

  ### Training Pong ###
  #####################
  # Main training loop
  while iteration < MAX_ITERS:

    tic = time.time()

    # # NON-PARALLEL: RL agent algorithm. By default, uses serial batch processing.
    # memories = collect_rollout(batch_size, game, model, getAction)
    # PARALLEL: RL agent algorithm. By default, uses serial batch processing.
    memories = parallelized_collect_rollout(batch_size, games, model, getAction)

    # Aggregate memories from multiple batches
    batch_memory = aggregate_memories(memories)

    # Come up with measures for tracking training (Ex: avg reward over time and average time to find hider)
    # # Track performance based on win percentage (calculated from rewards)
    # total_wins = sum(np.array(batch_memory.rewards) == 1)
    # total_games = sum(np.abs(np.array(batch_memory.rewards)))
    # win_rate = total_wins / total_games
    # smoothed_reward.append(100 * win_rate)
    mean_length = len(batch_memory.actions) / batch_size
    mean_reward = sum(batch_memory.rewards) / batch_size
    
    # Training!
    loss = train_step(
        model,
        optimizer,
        observations = np.stack(batch_memory.observations, 0),
        actions = np.array(batch_memory.actions),
        discounted_rewards = discount_rewards(batch_memory.rewards,batch_memory.game_ended)
    )
      
    iteration += 1 # Mark next episode

    toc = time.time()

    print("Finished Episode {}: runtime = {} , loss = {} , mean_reward = {} , mean length = {}".format(iteration, toc-tic, loss, mean_reward , mean_length))

    # Add something to save model once a certain iteration is hit, say every 100