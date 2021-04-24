'''
Created on Apr 14, 2021

@author: young
'''

# Imports
################################################################################

# python packages
import turtle
import time
import sys
import numpy as np
import math

# local files
import board_generator


# Parameters for Visualization
################################################################################
square_size = 15
squeeze_ratio = square_size / 24


# Primary Export/Functionality
################################################################################

class Visualization:
  '''
  Class for representing a game of Hide & Seek. This class will contain the
  environment and agents necessary for carrying out a game of Hide & Seek.
  Within this class are the various methods for each of the parts of the game.
  '''

  def __init__(self, environment, hiding_agent, seeking_agent, sleep_time):
    '''
    Initializes new Hiding Agent instance.
    '''
    # set the environment and agents for the visualizatoin
    self.environment = environment
    self.hiding_agent = hiding_agent
    self.seeking_agent = seeking_agent

    # save other parameters
    self.sleep_time = sleep_time

    # initialize visualization screen
    self.middle_position = self.environment.getMiddlePos()
    self.screen_width = self.environment.board.shape[0] * square_size * 1.1
    self.screen_height = self.environment.board.shape[1] * square_size * 1.1
    self.screen = turtle.Screen()
    self.screen.bgcolor("white")
    self.screen.setup(self.screen_width, self.screen_height)

    # initialize turtles for visualization
    self.wall = Wall()
    self.hider = Hider()
    self.seeker = Seeker()

    # initialize board
    turtle.tracer(0,0)
    for i in range(self.environment.board.shape[0]):
      for j in range(self.environment.board.shape[1]):
        if (self.environment.board[i,j]):
          x , y = self.index2Position(i,j)
          self.wall.goto(x,y)
          self.wall.stamp()

    hider_x , hider_y = self.index2Position(self.hiding_agent.position[0],self.hiding_agent.position[1])
    self.hider.goto(hider_x, hider_y)

    seeker_x , seeker_y = self.index2Position(self.seeking_agent.position[0],self.seeking_agent.position[1])
    self.seeker.goto(seeker_x, seeker_y)

    # update screen
    turtle.update()
    turtle.tracer(1,1)


  def moveHider(self):
    self.hider.move(self.index2Position(self.hiding_agent.position[0],self.hiding_agent.position[1]))
    time.sleep(self.sleep_time)

  def moveSeeker(self):
    self.seeker.move(self.index2Position(self.seeking_agent.position[0],self.seeking_agent.position[1]))
    time.sleep(self.sleep_time)

  def index2Position(self,i,j):
    middle_pos = self.environment.getMiddlePos()
    return ( (i-self.middle_position[0])*square_size , (j-self.middle_position[1])*square_size )

  def endVis(self):
    self.screen.exitonclick()
    sys.exit()

    


# Helper Classes
################################################################################
class Wall(turtle.Turtle):               # define a Maze class
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")            # the turtle shape
        self.color("black")             # color of the turtle
        self.shapesize(squeeze_ratio, squeeze_ratio, squeeze_ratio)
        self.penup()                    # lift up the pen so it do not leave a trail

# class for the End marker turtle (green square)
class Hider(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("circle")
        self.color("green")
        self.shapesize(squeeze_ratio, squeeze_ratio, squeeze_ratio)
        self.setheading(90)  # point turtle to point up
        self.penup()
    def move(self, position):
      self.goto(position[0],position[1])

# class for the sprite turtle (red turtle)
class Seeker(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("circle")
        self.color("red")
        self.shapesize(squeeze_ratio, squeeze_ratio, squeeze_ratio)
        self.setheading(90)  # point turtle to point up
        self.penup()
    def move(self, position):
      self.goto(position[0],position[1])
  
    
