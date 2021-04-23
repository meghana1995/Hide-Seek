'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################

# python packages
import heapq
import math
import numpy as np

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

# inverse actions
INVERSE_ACTIONS = {
  "right": "left",
  "left":   "right",
  "up":     "down",
  "down":   "up",
  "nothing": "nothing",
}


# Primary Export/Functionality
################################################################################

class Agent:
  '''
  Class for representing a generic Agent in the Hide & Seek game. This will
  be extended to create both our Hiding and Seeking Agents.
  '''

  def __init__(self, algorithm, env_shape, start_pos, vision_range):
    '''
    Initializes new Agent instance.
    '''
    # store agent's algorithm of choice
    self.algorithm = algorithm
    # initialize the agent's belief state of the environment as unknown
    # -1 == unknown , 0 == open , 1 == wall
    self.environment = np.full(env_shape, -1, np.int)
    self.visibility_table = {}
    # initialize agent's start position and label it as open
    self.start_position = start_pos
    self.position = start_pos
    self.environment[start_pos[0]][start_pos[1]] = 0
    # initialize agent's vision range
    self.vision_range = vision_range
    # initialize agent's plan as None
    self.plan = None
    # initialize set of positions visited by the agent
    self.visited_positions = set([start_pos])
    # initialize past moves (used for backtracking)
    self.current_path = []

  def resetState(self):
    '''
    Resets the agent's belief state back to its original state.
    '''
    self.environment.fill(-1)
    self.environment[self.start_position] = 0
    self.position = self.start_position
    self.plan = None
    self.visited_positions = set([self.start_position])
    self.current_path = []

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

  def randomAction(self):
    '''
    Returns a random valid action
    '''
    valid_actions = self.validActions()
    random_index = math.floor(np.random.random()*len(valid_actions))
    return valid_actions[random_index]

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
      visited.add(current["position"])
      # if we reached the goal, then return its path
      if (current["position"] == goal):
        return current["path"]
      # extend path to each of this positions neighbors
      for action, change in ALLOWED_ACTIONS.items():
        # get new position for action
        new_position = ( 
          current["position"][0] + change[0],
          current["position"][1] + change[1],
        )
        # add position to queue if valid and not already visited
        if (new_position not in visited and self.environment[new_position] == 0):
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

  def dFS(self):
    '''
    Uses online version of DFS with backtracking to allow agent to wander
    randomly around the board.
    '''
    # init list for storing valid actions
    valid_actions = []
    # iterate over actions and test if it results in valid position
    for action, change in MOVE_ACTIONS.items():
      new_position = ( 
        self.position[0] + change[0],
        self.position[1] + change[1],
      )
      if (self.environment[new_position] == 0 and new_position not in self.visited_positions):
        valid_actions.append(action)
    # if valid actions is empty then backtrack, else pick a random action
    if (len(valid_actions) == 0):
      return INVERSE_ACTIONS[self.current_path[-1]]
    else:
      random_index = math.floor(np.random.random()*len(valid_actions))
      return valid_actions[random_index]

  def hC(self):
    '''
    Uses online version of DFS with backtracking to allow agent to wander
    randomly around the board.
    '''
    # init best action for storing valid actions
    best_action = None
    best_score = -1
    # iterate over actions and test if it results in valid position
    for action, change in MOVE_ACTIONS.items():
      new_position = ( 
        self.position[0] + change[0],
        self.position[1] + change[1],
      )
      if (self.environment[new_position] == 0):
        hideability = self.hidabilityScore(new_position)
        if (hideability > best_score):
          best_score = hideability
          best_action = action
    # return best action
    return best_action

  def improvedHC(self):
    '''
    Uses online version of DFS with backtracking to allow agent to wander
    randomly around the board.
    '''
    # init best action for storing valid actions
    best_action = None
    best_score = -1
    # iterate over actions and test if it results in valid position
    for action, change in MOVE_ACTIONS.items():
      new_position = ( 
        self.position[0] + change[0],
        self.position[1] + change[1],
      )
      if (self.environment[new_position] == 0 and new_position not in self.visited_positions):
        hideability = self.hidabilityScore(new_position)
        if (hideability > best_score):
          best_score = hideability
          best_action = action
    # if no valid actions then backtrack, else pick highest value action
    if (best_action == None):
      return INVERSE_ACTIONS[self.current_path[-1]]
    else:
      return best_action

  def randomHC(self):
    '''
    Uses online version of DFS with backtracking to allow agent to wander
    randomly around the board.
    '''
    # init lists for storing valid actions and scores
    valid_actions = []
    action_scores = []
    # iterate over actions and test if it results in valid position
    for action, change in MOVE_ACTIONS.items():
      new_position = ( 
        self.position[0] + change[0],
        self.position[1] + change[1],
      )
      if (self.environment[new_position] == 0 and new_position not in self.visited_positions):
        hideability = self.hidabilityScore(new_position)
        valid_actions.append(action)
        action_scores.append(hideability)
    # if no valid actions then backtrack, else pick random action based on scores
    if (len(valid_actions) == 0):
      return INVERSE_ACTIONS[self.current_path[-1]]
    else:
      action_probs = helper_util.normalize(action_scores)
      return np.random.choice(valid_actions,p=action_probs)

  def hideability(self, position):
    '''
    Hueristic function used by agents to estimate the hideability value of a
    given tile, based off currently know knowledge.
    '''
    # weights for combining the scores
    h_weight = .5
    v_weight = 0
    d_weight = .5
    # grab scores
    h_score = self.hidabilityScore(position)
    v_score = self.visibilityScore(position)
    d_score = self.distanceScore(position)
    # get combined score
    return h_weight*h_score + v_weight*v_score + d_weight*d_score

  def hidabilityScore(self,position):
    '''
    Score that rates how good of a hiding spot the given position is based off
    current knowledge of how many positions are visible from it. Used as one
    component in the hideability heuristic. Designed to fall in range [0,1].
    '''
    # grab the largest number of tiles visible from any given spot
    max_visible = 2*(self.vision_range**2 + self.vision_range)
    # number of positions visible from given position
    visible = len(self.visibility_table[position])
    # return ratio
    return (max_visible - visible) / (max_visible - 1)

  def visibilityScore(self,position):
    '''
    Score that rates how visible other positions are from the given position.
    Can be thought of as the inverse to the hideability score. Designed to fall
    in range [0,1].
    '''
    # grab the largest number of tiles visible from any given spot
    max_visible = 2*(self.vision_range**2 + self.vision_range)
    # number of positions visible from given position
    visible = len(self.visibility_table[position])
    # return ratio
    return (visible-1) / (max_visible-1)


  def distanceScore(self,position):
    '''
    Score that rates how far the position is from the starting position in terms
    of its Manhattan distance. Designed to fall in range [0,1].
    '''
    # max distance possible from starting position
    max_distance = (self.environment.shape[0] - 1) / 2
    # Manhattan distance from start
    distance = helper_util.ManhattanDist(position,self.start_position)
    # return ratio
    return distance / max_distance
    




# Helper Functions
################################################################################      


# Unit Tests
################################################################################
if __name__ == "__main__":

  # create test agent
  test_agent = Agent((11,11), (5,5), 3)

  # print environment
  print(test_agent.environment)


    
