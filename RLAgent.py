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


# Constants
################################################################################

# allowed actions
ALLOWED_ACTIONS = {
  "right":  (1,0),
  "left":   (-1,0),
  "up":     (0,1),
  "down":   (0,-1),
  "nothing": (0,0),
}

# movement actions 
MOVE_ACTIONS = {
  "right":  (1,0),
  "left":   (-1,0),
  "up":     (0,1),
  "down":   (0,-1),
}

# movement actions 
MOVE_ACTIONS_LIST = [
  "right",
  "left",
  "up",
  "down",
]


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

  def __init__(self, env_shape, start_pos, vision_range, model):
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

    # save model used by the agent
    self.model = model    

    # store initial position of the seeker
    self.position = start_pos

    # store position of hider (None until hider perceived)
    self.hider_position = None

    # initialize agent's internal state of the environment
    # this is used for determining actions and stores 2 matrices:
    #  1) What the agent has observed in each position (0 == unknown , 1 == open , 2 == wall , 3 == Agent)   # May need to change this
    #  2) Estimate of # of tiles visible from each position (0-24)
    self.env_shape
    self.initAgentState(env_shape)

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
    self.visibility_table = {}
    # build matrix storing types of tiles agent has observed at each positon
    # 0 == unknown , 1 == open , 2 == wall , 3 == Agent (may need to change these)
    self.state = np.zeros(self.env_shape)
    self.state[self.start_position] = 3

    # # below was when I wanted to also give the agent the visibility scores, but now I'm not doig that
    # # build matrix storing estimated visibility score for each position
    # visibility = np.zeros(self.env_shape)
    # # combine matrices into single tensor stored in agent
    # self.state = np.stack([tiles,visibility],axis=-1)

  def updateState(self, open_squares, wall_squares, visibilityTable, hider_position):
    '''
    This function allows the agent to update its stored state of the environment
    based off the percepts it receives. For the RL Agent, this function returns
    the number of tiles previously not observed in order to determine a reward
    for training.
    '''
    # update hider position
    self.hider_position = hider_position

    # # update visibility table for current seeker position
    # self.updateVisTable(self.position, open_squares)
    # self.state[self.position][1] = self.visibility_table[self.position]

    # init variable for counting newly observed positions
    new_positions = 0

    # update state for observed open squares
    for position in open_squares:
      if self.state[position] == 0:
        new_positions += 1
      self.state[position] = 1
      # self.updateVisTable(position, visibilityTable[position])
      # self.state[position][1] = self.visibility_table[position]

    # update state for observed walls
    for position in wall_squares:
      if self.state[position] == 0:
        new_positions += 1
      self.state[position][0] = 2

    # return number of new positions observed
    return new_positions

  # not used anymore, but keeping in case that changes
  def updateVisTable(self, position, open_squares):
    '''
    Update the visibility table for the agent using the given set of
    visbile open squares and the position to be updated.
    '''
    if (position in self.visibility_table):
      self.visibility_table[position] = self.visibility_table[position].union(open_squares)
    else:
      self.visibility_table[position] = open_squares

  def getAction(self):
    '''
    This funciton represents the Seeking Agent determining what action to carry
    out next while trying to find the Hiding Agent.
    '''
    # if agent has plan then follow this plan
    if (self.plan is not None):
      return self.plan.pop(0)
    # else if hider position is known build a plan and execute first action
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
    if (self.state[new_position] == 1):
      self.state[self.position] = 1
      self.state[new_position] = 3
      self.position = new_position

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
    return self.state[new_position] == 1

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
      if (self.state[new_position] == 1):
        valid_actions.append(action)
    # return valid actions
    return valid_actions

  def modelAction(self):
    # add batch dimension to the observation if only a single example was provided
    observation = np.expand_dims(self.state, axis=0)

    # grab probabilites for each action by passing observation into model
    logits = model.predict(observation)
    
    # select action based off logits above
    action = tf.random.categorical(logits, num_samples=1)
    action = action.numpy().flatten()[0]

    # return action as string value from MOVE_ACTIONS_LIST
    return MOVE_ACTIONS_LIST[action]



# Convolutional Model for Seeker Agent
################################################################################

# Functionally define layers for convenience
# All convolutional layers will have ReLu activation
Embedding = tf.keras.layers.Embedding
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
BatchNorm = tf.keras.layers.BatchNormalization
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

def createModel():
  model = tf.keras.models.Sequential([

    # Embedding layer to convert integers to dense vector space
    # Embedding(3,2),

    # Convolutional layers
    Conv2D(filters=32, kernel_size=5, strides=2),
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


# Memory of Agent For Training
################################################################################
class Memory:
  def __init__(self): 
      self.clear()

  # Resets/restarts the memory buffer
  def clear(self): 
      self.observations = []
      self.actions = []
      self.rewards = []
      self.game_ended = []

  # Add observations, actions, rewards to memory
  def add_to_memory(self, new_observation, new_action, new_reward, game_ended): 
      self.observations.append(new_observation)
      self.actions.append(new_action)
      self.rewards.append(new_reward)
      self.game_ended.append(game_ended)

# Helper function to combine a list of Memory objects into a single Memory.
#     This will be very useful for batching.
def aggregate_memories(memories):
  batch_memory = Memory()
  
  for memory in memories:
    for step in zip(memory.observations, memory.actions, memory.rewards, memory.game_ended):
      batch_memory.add_to_memory(*step)
  
  return batch_memory


# Reward and Discounted Reward Functions
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
def discount_rewards(rewards, game_ended, gamma=0.99): 
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
      # Reset the sum if the game ended (will be True value in array if that is the case)
      if game_ended[t]:
        R = 0
      # update the total discounted reward as before
      R = R * gamma + rewards[t]
      discounted_rewards[t] = R
      
  return normalize(discounted_rewards)

def rewardFunc(new_positions):
  return new_positions*.1 - 1


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
#   game: game of hide and seek
#   model: Pong agent model
#   choose_action: choose_action function
# Returns:
#   memories: array of Memory buffers, of length batch_size, corresponding to the
#     episode executions from the rollout
def collect_rollout(batch_size, game, model, choose_action):

  # reset the game (i.e. build new environment board for batches)
  # this means all samples in batch use the same board
  game.resetEnv()

  # Holder array for the Memory buffers
  memories = []

  # Process batches serially by iterating through them
  for b in range(batch_size):

    # Instantiate Memory buffer, restart the environment, and grab observation
    memory = Memory()

    # reset the hide and seek game, and run the hider process
    game.resetAgents()
    game.runHiderSequence()

    # let agent perceive environment initially
    open , walls , visibilityTable = game.environment.perceiveEnv(game.seeking_agent)
    hider_position = game.foundHider(open,walls,game.hiding_agent.position)
    game.seeking_agent.updateState(open,walls,visibilityTable,hider_position)

    # grab current state of the seeker as current observation
    current_observation = game.seeker.state.copy()
    next_observation = current_observation

    while hider_position == None:

      # get next action based on observed change to environment
      action = choose_action(model, current_observation)
      
      # Take the chosen action
      game.seeking_agent.performAction(MOVE_ACTIONS_LIST[action])

      # Allow agent to perceive environment
      open , walls , visibilityTable = game.environment.perceiveEnv(game.seeking_agent)
      hider_position = game.foundHider(open,walls,game.hiding_agent.position)
      new_positions = game.seeking_agent.updateState(open,walls,visibilityTable,hider_position)

      # grab reward from previous action
      reward = rewardFunc(new_positions)

      # save the observed state, the action that was taken, the resulting reward, and state of the game (over or not)
      memory.add_to_memory(current_observation, action, reward, hider_position == None)

      # update observation
      current_observation = game.seeker.state.copy()
    
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
  # Parameters for Env. and Hiding Agent
  hiding_time = 400
  vision_range = 3
  hiding_alg = "rhc"
  hider_weights = (1/3,0,2/3)
  h_randomness = .5
  
  # Hyperparameters for Training
  initial_learning_rate = .1
  decay_steps = 10
  decay_rate = .99
  MAX_ITERS = 10 # increase the maximum to train longer
  batch_size = 1 # number of batches to run

  # Initialize Environment
  environment = Environment(vision_range)
  env_shape = environment.board.shape
  middle_pos = environment.getMiddlePos()

  # Initialize Agents (Including RL Model in Seeking Agent)
  hiding_agent = HidingAgent(hiding_alg,env_shape,middle_pos,vision_range,hider_weights,h_randomness,hiding_time)
  model = createModel()
  seeking_agent = RLAgent(env_shape,middle_pos,vision_range, model)

  # Optimizer
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  iteration = 0 # counter for training steps

  # Initialize Hide and Seek Game
  game = HideAndSeek(environment,hiding_agent,seeking_agent,hiding_time)

  ### Training Pong ###
  #####################
  # Main training loop
  while iteration < MAX_ITERS:

    tic = time.time()

    # RL agent algorithm. By default, uses serial batch processing.
    memories = collect_rollout(batch_size, game, model, getAction)

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

    toc = time.time()

    print("Finished Episode {}: runtime = {}".format(iteration,toc-tic))

    # Add something to save model once a certain iteration is hit, say every 100