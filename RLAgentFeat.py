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
import heapq

# local files
import helper_util


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

    # save model used by the agent
    self.input_shape = [env_shape[0],env_shape[1],3]
    self.model = model   

    # initialize the state of the agents memory by doing a resetState call
    self.resetState() 

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
    # note that different axis are used for different ffeatures
    #   Axis 0 - Known vs Unknown:  0 = Unknown   ,   1 = Known
    #   Axis 1 - Open vs Wall:      0 = Open      ,   1 = Wall
    #   Axis 2 - Seeker Position:   0 = No Seeker ,   1 = Seeker
    self.state = np.zeros(self.input_shape)
    self.state[self.start_position][0] = 1
    self.state[self.start_position][2] = 1

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
      if self.state[position][0] == 0:
        new_positions += 1
        self.state[position][0] = 1
      # self.updateVisTable(position, visibilityTable[position])
      # self.state[position][1] = self.visibility_table[position]

    # update state for observed walls
    for position in wall_squares:
      if self.state[position][0] == 0:
        new_positions += 1
        self.state[position][0] = 1
        self.state[position][1] = 1

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
    if (self.state[new_position][1] == 0):
      self.state[self.position][2] = 0
      self.state[new_position][2] = 1
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
    return self.state[new_position][1] == 0

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
      if (self.state[new_position][1] == 0):
        valid_actions.append(action)
    # return valid actions
    return valid_actions

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
    # else use algorithm to pick next action
    else:
      # add batch dimension to the observation if only a single example was provided
      observation = np.expand_dims(self.state, axis=0)

      # grab probabilites for each action by passing observation into model
      logits = self.model.predict(observation)
      
      # select action based off logits above
      action = tf.random.categorical(logits, num_samples=1)
      action = action.numpy().flatten()[0]

      # return action as string value from MOVE_ACTIONS_LIST
      return MOVE_ACTIONS_LIST[action]

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
      # skip to next item on queue if this position has been visited
      if (current["position"] in visited):
        continue
      # else add to list of visited
      else:
        visited.add(current["position"])
      # if we reached the goal, then return its path
      if (current["position"] == goal):
        return current["path"]
      # extend path to each of this positions neighbors
      for action, change in MOVE_ACTIONS.items():
        # get new position for action
        new_position = ( 
          current["position"][0] + change[0],
          current["position"][1] + change[1],
        )
        # add position to queue if valid and not already visited
        if (new_position not in visited and self.state[new_position][1] == 0):
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



# Convolutional Model for Seeker Agent
################################################################################

# Functionally define layers for convenience
# All convolutional layers will have ReLu activation
Embedding = tf.keras.layers.Embedding
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
BatchNorm = tf.keras.layers.BatchNormalization
MaxPool2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout =  tf.keras.layers.Dropout

def createModel():
  model = tf.keras.models.Sequential([

    # Embedding layer to convert integers to dense vector space
    # Embedding(3,2),

    # Convolutional layers
    Conv2D(32, (3,3)),
    BatchNorm(),
    Conv2D(32, (3,3)),
    BatchNorm(),
    MaxPool2D((2, 2)),
    Conv2D(32, (3,3)),
    BatchNorm(),
    Conv2D(32, (3,3)),
    BatchNorm(),
    MaxPool2D((2, 2)),

    # Flatten before dense layers
    Flatten(),
    
    # Fully connected layers
    Dense(units=512, activation='relu'),
    BatchNorm(),
    Dense(units=512, activation='relu'),
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

def rewardFunc(new_positions, hider_position):
  reward = new_positions*.5 - 1
  if (hider_position != None):
    reward += 50
  return reward

# def rewardFunc(new_positions, hider_position):
#   if (hider_position != None):
#     return 100
#   else:
#     return 0


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
def collect_rollout(batch_size, game, model, choose_action, max_game_length=4000):

  # # reset the game (i.e. build new environment board for batches)
  # # this means all samples in batch use the same board
  # game.resetEnv()

  # Holder array for the Memory buffers
  memories = []

  # Process batches serially by iterating through them
  for b in range(batch_size):

    # Instantiate Memory buffer, restart the environment, and grab observation
    memory = Memory()

    # reset the hide and seek game, and run the hider process
    game.resetEnv()
    game.runHiderSequence()

    # let agent perceive environment initially
    open , walls , visibilityTable = game.environment.perceiveEnv(game.seeking_agent)
    hider_position = game.foundHider(open,walls,game.hiding_agent.position)
    game.seeking_agent.updateState(open,walls,visibilityTable,hider_position)

    # grab current state of the seeker as current observation
    current_observation = game.seeking_agent.state.copy()
    time.sleep(1)

    game_length = 0
    done = False
    while not done:

      # get next action based on observed change to environment
      action = choose_action(model, current_observation)
      
      # Take the chosen action
      game.seeking_agent.performAction(MOVE_ACTIONS_LIST[action])

      # Allow agent to perceive environment
      open , walls , visibilityTable = game.environment.perceiveEnv(game.seeking_agent)
      hider_position = game.foundHider(open,walls,game.hiding_agent.position)
      new_positions = game.seeking_agent.updateState(open,walls,visibilityTable,hider_position)

      # grab reward from previous action
      reward = rewardFunc(new_positions,hider_position)

      # determine if game should be stopped
      game_length += 1
      if (game_length >  max_game_length):
        done = True
      else:
        done = (hider_position != None)

      # save the observed state, the action that was taken, the resulting reward, and state of the game (over or not)
      memory.add_to_memory(current_observation, action, reward, done)

      # update observation
      current_observation = game.seeking_agent.state.copy()
      time.sleep(1)
    
    # Add the memory from this batch to the array of all Memory buffers
    memories.append(memory)
  
  return memories


# Parallel Rollout Function
################################################################################
def parallelized_collect_rollout(batch_size, games, model, choose_action, max_game_length=4000):

    assert len(games) == batch_size, "Number of parallel environments must be equal to the batch size."

    # Instantiate Memory buffer, restart the environment, and grab observation
    memories = [Memory() for _ in range(batch_size)]

    # initialize games
    current_observations = []
    for b in range(batch_size):
      # reset the hide and seek game, and run the hider process
      games[b].resetEnv()
      games[b].runHiderSequence()
      # let agent perceive environment initially
      open , walls , visibilityTable = games[b].environment.perceiveEnv(games[b].seeking_agent)
      hider_position = games[b].foundHider(open,walls,games[b].hiding_agent.position)
      games[b].seeking_agent.updateState(open,walls,visibilityTable,hider_position)
      # grab current state of the seeker as current observation
      current_observations.append(games[b].seeking_agent.state.copy())

    # Instantiate done, rewards, and game_length
    done = [False] * batch_size
    rewards = [0] * batch_size
    game_lengths = [0] * batch_size
    
    while True:

        # get next action based on observed change to environment
        actions_not_done = choose_action(model, np.stack(current_observations), single=False)

        actions = [None] * batch_size
        ind_not_done = 0
        for b in range(batch_size):
            if not done[b]:
                actions[b] = actions_not_done[ind_not_done]
                ind_not_done += 1

        for b in range(batch_size):
            if done[b]:
                continue
            games[b].seeking_agent.performAction(MOVE_ACTIONS_LIST[actions[b]])
            open , walls , visibilityTable = games[b].environment.perceiveEnv(games[b].seeking_agent)
            hider_position = games[b].foundHider(open,walls,games[b].hiding_agent.position)
            new_positions = games[b].seeking_agent.updateState(open,walls,visibilityTable,hider_position)
            rewards[b] = rewardFunc(new_positions,hider_position)
            # determine if game should be stopped
            game_lengths[b] += 1
            if (game_lengths[b] >  max_game_length):
              done[b] = True
            else:
              done[b] = (hider_position != None)
            memories[b].add_to_memory(current_observations[b], actions[b], rewards[b], done[b])
            # update observation
            current_observations[b] = games[b].seeking_agent.state.copy()

        if all(done):
            break

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

  # return the loss for analysis
  return loss

