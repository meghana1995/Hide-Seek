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



# Reinforcement Learning Agent
################################################################################
class RLAgent:
  '''
  Class for representing our Seeking Agent that uses Convolutional Neural
  Network Model to seach for the Hiding Agent during a game of Hide & Seek.
  This model includes methods for training the model using simulations of the
  Hide and Seek game and Reinforcement Learning for encouraging optimal search
  behaviors.
  '''

  def __init__(self, env_shape, start_pos, vision_range, model_path=None):
    '''
    Initializes new RL Seeking Agent instance. If model_path is given, then the
    agent is initialized using the saved model in the directory specified by the
    model_path. Otherwise, a new untrained model is used.
    '''
    # store the inputs since they may be used later (such as resetting the agent)
    self.env_shape = env_shape
    self.start_position = start_pos
    self.vision_range = vision_range

    # initialize the state of the agents memory by doing a resetState call
    self.resetState()

    # initialize the neural network model for the agent or load saved modal
    if (model_path != None):
      self.model = createModel()
    else:
      self.model = createModel()
    

    # store initial position of the seeker
    self.position = start_pos

    # store position of hider (None until hider perceived)
    self.hider_position = None

    # initialize agent's internal state of the environment
    # this is used for determining actions and stores 2 matrices:
    #  1) What the agent has observed in each position (-1 == unknown , 0 == open , 1 == wall , 2 == Agent)   # May need to change this
    #  2) Estimate of # of tiles visible from each position (0-24)
    self.env_shape
    self.initAgentState(env_shape)

    # initialize the agent's belief state of the environment as unknown
    # -1 == unknown , 0 == open , 1 == wall
    self.environment = np.full(env_shape, -1, np.int)
    self.visibility_table = {}
    # initialize agent's start position and label it as open
    self.start_position = start_pos
    self.position = start_pos
    self.environment[start_pos[0]][start_pos[1]] = 0
    # store agent's vision range
    self.vision_range = vision_range
    # store weights for heuristic function
    self.h_weights = h_weights
    # initialize agent's plan as None
    self.plan = None
    # initialize set of positions visited by the agent
    self.visited_positions = set([start_pos])
    # initialize past moves (used for backtracking)
    self.current_path = []
    # store randomness of the agent
    self.randomness = randomness

  def resetState(self):
    '''
    Initializes or resets the agent's memory state back to its original state.
    This sets all the data structures and values the agent needs to model its
    environment and determine next actions.
    '''
    # set initial position of the seeker
    self.position = self.start_position

    # set initial position of hider (None until hider perceived)
    self.hider_position = None

    # set plan agent stores for getting to hider (None until hider perceived)
    self.plan = None

    # set agents understanding of what it has percieved in the environment
    # empty visibility table storing open positions visible from open position
    


    self.environment.fill(-1)
    self.environment[self.start_position] = 0
    self.position = self.start_position
    self.plan = None
    self.visited_positions = set([self.start_position])
    self.current_path = []
    self.hider_position = None


  def updateState(self, open_squares, wall_squares, visibilityTable, hider_position):
    '''
    This function allows the agent to update its belief state of the environment
    based off the percepts it receives.
    '''
    # call super to update environment based off visible squares
    super().updateState(open_squares, wall_squares, visibilityTable)
    # update game clock
    self.hider_position = hider_position

  def getAction(self):
    '''
    This funciton represents the Seeking Agent determining what action to carry
    out next while trying to find the Hiding Agent.
    '''
    # if agent has plan then follow this plan
    if (self.plan is not None):
      return self.plan.pop(0)
    # else if hider position is know build a plan and execute first action
    elif (self.hider_position is not None):
      self.plan = self.aStar(self.hider_position)
      return self.plan.pop(0)
    # else CNN decides next action
    else:
      # TODO
      # replace this with neural networks getAction
      return "nothing"

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
    if (self.environment[new_position] == 0):
      self.position = new_position
      self.visited_positions.add(new_position)
      # consider action to be backtracking if it was inverse of last action
      if (len(self.current_path)>0):
        inverse_last_action = INVERSE_ACTIONS[self.current_path[-1]]
      else:
        inverse_last_action = "none"
      if (action == inverse_last_action):
        del self.current_path[-1]
      else:
        self.current_path.append(action)

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

  def updateVisTable(self, position, open_squares):
    '''
    Update the visibility table for the agent using the given set of
    visbile open squares and the position to be updated.
    '''
    if (position in self.visibility_table):
      self.visibility_table[position] = self.visibility_table[position].union(open_squares)
    else:
      self.visibility_table[position] = open_squares

  def updateState(self, open_squares, wall_squares, visibilityTable):
    '''
    This function allows the agent to update its belief state of the environment
    based off the open and wall squares it is able to perceive.
    '''
    # update visibility table for current position
    self.updateVisTable(self.position, open_squares)
    # update environment and visibility table for open squares
    for square in open_squares:
      self.environment[square] = 0
      self.updateVisTable(square, visibilityTable[square])
    # update environment for wall squares
    for square in wall_squares:
      self.environment[square] = 1



# Convolutional Model for Seeker Agent
################################################################################

# Functionally define layers for convenience
# All convolutional layers will have ReLu activation
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
BatchNorm = tf.keras.layers.BatchNormalization
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

def createModel():
  model = tf.keras.models.Sequential([

    # Convolutional layers
    Conv2D(filters=32, kernel_size=5, strides=2,  input_shape = (51,51,2)),
    BatchNorm(),
    Conv2D(filters=48, kernel_size=5, strides=2),
    BatchNorm(),
    Conv2D(filters=64, kernel_size=3, strides=2),
    BatchNorm(),
    Conv2D(filters=64, kernel_size=3, strides=2),
    BatchNorm(),

    # Flatten before dense layers
    Flatten(),
    
    # Fully connected layers
    Dense(units=128, activation='relu'),
    BatchNorm(),
    Dense(units=64, activation='relu'),
    BatchNorm(),

    # Output Layer
    Dense(units=4, activation=None)
  
  ])
  return model


# Grabbing an Action from Model
################################################################################
def getAction(model, observations, single=True):
  # add batch dimension to the observation if only a single example was provided
  observation = np.expand_dims(observations, axis=0) if single else observations

  # grab probabilites for each action by passing observation into model
  logits = model.predict(observation)
  
  # select action based of logits above
  action = tf.random.categorical(logits, num_samples=1)
  
  action = action.numpy().flatten()

  return action[0] if single else action


# Memeory of Agent
################################################################################
class Memory:
  def __init__(self): 
      self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
      self.observations = []
      self.actions = []
      self.rewards = []

  # Add observations, actions, rewards to memory
  def add_to_memory(self, new_observation, new_action, new_reward): 
      self.observations.append(new_observation)
      self.actions.append(new_action)
      self.rewards.append(new_reward)

# Helper function to combine a list of Memory objects into a single Memory.
#     This will be very useful for batching.
def aggregate_memories(memories):
  batch_memory = Memory()
  
  for memory in memories:
    for step in zip(memory.observations, memory.actions, memory.rewards):
      batch_memory.add_to_memory(*step)
  
  return batch_memory


# Reward Function
################################################################################
def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  return x.astype(np.float32)

# Compute normalized, discounted rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor.
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, game_ended=False, gamma=0.99): 
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
      # Reset the sum if the game ended
      if game_ended:
        R = 0
      # update the total discounted reward as before
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R
      
  return normalize(discounted_rewards)


# Rollout Function
################################################################################
# Key steps for agent's operation in the environment, until completion of a rollout.
# An observation is drawn; the agent (controlled by model) selects an action;
#   the agent executes that action in the environment and collects rewards;
#   information is added to memory.
# This is repeated until the completion of the rollout -- the H&S game ends.
# Processes multiple batches serially.
#
# Arguments:
#   batch_size: number of batches, to be processed serially
#   env: environment
#   model: Pong agent model
#   choose_action: choose_action function
# Returns:
#   memories: array of Memory buffers, of length batch_size, corresponding to the
#     episode executions from the rollout
def collect_rollout(batch_size, env, model, choose_action):

  # Holder array for the Memory buffers
  memories = []

  # Process batches serially by iterating through them
  for b in range(batch_size):

    # Instantiate Memory buffer, restart the environment
    memory = Memory()
    current_observation = env.reset()
    next_observation = current_observation
    done = False # tracks whether the episode (game) is done or not

    while not done:

      # get next action based on observed change to environment
      action = choose_action(model, current_observation)
      
      # Take the chosen action
      # TODO 
      next_observation, reward, done, info = env.step(action)

      # save the observed frame difference, the action that was taken, and the resulting reward
      memory.add_to_memory(current_observation, action, reward) # TODO

      # update observation
      current_observation = next_observation
    
    # Add the memory from this batch to the array of all Memory buffers
    memories.append(memory)
  
  return memories


# Loss Function
################################################################################
# Arguments:
#   logits: network's predictions for actions to take
#   actions: the actions the agent took in an episode
#   rewards: the rewards the agent received in an episode
# Returns:
#   loss
def compute_loss(logits, actions, rewards): 
  # compute the negative log probabilities
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=actions)
  
  # scale the negative log probability by the rewards
  loss = tf.reduce_mean( neg_logprob * rewards )

  return loss


# Training Step
################################################################################
def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
      # Forward propagate through the agent network
      logits = model(observations)

      # call the compute_loss function to compute the loss
      loss = compute_loss(logits, actions, discounted_rewards)

  # run backpropagation to minimize the loss using the tape.gradient method
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))



# Main - (For Training and Saving Model)
################################################################################
if __name__ == "__main__":
  ### Hyperparameters and setup for training ###
  ##############################################
  # Hyperparameters
  learning_rate = 1e-3
  MAX_ITERS = 10 # increase the maximum to train longer
  batch_size = 10 # number of batches to run

  # Model, optimizer
  model = createModel()
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  iteration = 0 # counter for training steps

  # Batches and environment
  # To parallelize batches, we need to make multiple copies of the environment.
  # TODO
  envs = [create_pong_env() for _ in range(batch_size)] # For parallelization

  ### Training Pong ###
  #####################
  # Main training loop
  while iteration < MAX_ITERS:

    tic = time.time()

    # RL agent algorithm. By default, uses serial batch processing.
    # TODO
    memories = collect_rollout(batch_size, env, model, getAction)

    toc = time.time()

    # Aggregate memories from multiple batches
    batch_memory = aggregate_memories(memories)

    # Come up with measures for tracking training (Ex: avg reward over time and average time to find hider)
    # # Track performance based on win percentage (calculated from rewards)
    # total_wins = sum(np.array(batch_memory.rewards) == 1)
    # total_games = sum(np.abs(np.array(batch_memory.rewards)))
    # win_rate = total_wins / total_games
    # smoothed_reward.append(100 * win_rate)
    
    # Training!
    train_step(
        model,
        optimizer,
        observations = np.stack(batch_memory.observations, 0),
        actions = np.array(batch_memory.actions),
        discounted_rewards = discount_rewards(batch_memory.rewards)
    )
      
    iteration += 1 # Mark next episode

    # Add something to save model once a certain iteration is hit, say every 100