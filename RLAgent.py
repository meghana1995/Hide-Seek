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
  MAX_ITERS = 100 # increase the maximum to train longer
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