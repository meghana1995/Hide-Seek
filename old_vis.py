'''
Created on Mar 31, 2021

@author: kmegh
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


# Parameters for Grid
################################################################################
radius = 50
square_size = 7
hider_start = (30,30)
seeker_start = (0,0)
# seed = 1
seed = math.floor(np.random.rand()*456132545)
simplex_cutoffs = ( .67 , .67 )
simplex_scales = ( 10 , 5 )
opening_width = 3
second_pass = True


# Parameters for Visualization
################################################################################
width = radius*2 + 1
screen_size = width * square_size
squeeze_ratio = square_size / 24


# Generate Grid
################################################################################ 
grid = board_generator.generateBoard(
  radius, seed, simplex_cutoffs, simplex_scales, opening_width, second_pass
)


# Functions for transforming from position to board (x,y) position
def index2Position(i,j):
    return ( (i-radius)*square_size , (radius-j)*square_size )
            
def loc2Position(i,j):
    return ( i*square_size , -j*square_size )
            


wn = turtle.Screen()               # define the turtle screen
wn.bgcolor("white")                # set the background colour
wn.setup(screen_size*1.1,screen_size*1.1)                 # setup the dimensions of the working window


# class for the Maze turtle (white square)
class Maze(turtle.Turtle):               # define a Maze class
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")            # the turtle shape
        self.color("black")             # color of the turtle
        self.shapesize(squeeze_ratio, squeeze_ratio, squeeze_ratio)
        self.penup()                    # lift up the pen so it do not leave a trail
        self.speed(0)                   # sets the speed that the maze is written to the screen

# class for the End marker turtle (green square)
class End(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("green")
        self.shapesize(squeeze_ratio, squeeze_ratio, squeeze_ratio)
        self.penup()
        self.speed(0)

# class for the sprite turtle (red turtle)
class sprite(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("turtle")
        self.color("red")
        self.shapesize(squeeze_ratio, squeeze_ratio, squeeze_ratio)
        self.setheading(270)  # point turtle to point down
        self.penup()
        self.speed(0)


    def spriteDown(self):
        if (self.heading() == 270):                   # check to see if the sprite is pointing down
            x_walls = round(sprite.xcor(),0)          # sprite x coordinates =
            y_walls = round(sprite.ycor(),0) 
            if (x_walls, y_walls) in finish:          # if sprite and the
                print("Finished")
                endProgram()
            if (x_walls +square_size, y_walls) in walls:          # check to see if they are walls on the left
                if(x_walls, y_walls -square_size) not in walls:   # check to see if path ahead is clear
                    self.forward(square_size)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(square_size)


    def spriteleft(self):
        if (self.heading() == 0):
            x_walls = round(sprite.xcor(),0)
            y_walls = round(sprite.ycor(),0)
            if (x_walls, y_walls) in finish:   # check turtle coordinates are at the finish line
                print("Finished")
                endProgram()
            if (x_walls, y_walls +square_size) in walls:       # check to see if they are walls on the left
                if(x_walls +square_size, y_walls) not in walls:
                    self.forward(square_size)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(square_size)


    def spriteUp(self):
        if (self.heading() == 90):
            x_walls = round(sprite.xcor(),0)
            y_walls = round(sprite.ycor(),0)
            if (x_walls, y_walls) in finish:   # check turtle coordinates are at the finish line
                print("Finished")
                endProgram()
            if (x_walls -square_size, y_walls ) in walls:  # check to see if they are walls on the left
                if (x_walls, y_walls + square_size) not in walls:
                    self.forward(square_size)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(square_size)

    def spriteRight(self):
        if (self.heading() == 180):

            x_walls = round(sprite.xcor(),0)
            y_walls = round(sprite.ycor(),0)
            if (x_walls, y_walls) in finish:   # check turtle coordinates are at the finish line
                print("Finished")
                endProgram()
            if (x_walls, y_walls -square_size) in walls:  # check to see if they are walls on the left
                if (x_walls - square_size, y_walls) not in walls:
                    self.forward(square_size)
                else:
                    self.right(90)
            else:
                self.left(90)
                self.forward(square_size)


def endProgram():
    wn.exitonclick()
    sys.exit()


def setupMaze(grid):
    
    for y in range(len(grid)):                       # select each line in the grid
        for x in range(len(grid[y])):                # identify each character in the line
            isWall = grid[y][x]                   # assign the grid reference to the variable character
            screen_x , screen_y = index2Position(x,y)              # assign screen_y to screen starting position for y ie  288

            if isWall:                     # if grid character contains an +
                maze.goto(screen_x, screen_y)        # move turtle to the x and y location and
                maze.stamp()                         # stamp a copy of the turtle (white square) on the screen
                walls.append((screen_x, screen_y))   # add coordinate to walls list
    
    hider_x , hider_y = loc2Position(hider_start[0], hider_start[1])
    end.goto(hider_x, hider_y)         # move turtle to the x and y location and
    end.stamp()                          # stamp a copy of the turtle (green square) on the screen
    finish.append((hider_x, hider_y))  # add coordinate to finish list
    
    seeker_x , seeker_y = loc2Position(seeker_start[0], seeker_start[1])
    sprite.goto(seeker_x, seeker_y)      # move turtle to the x and y location
    
    


# ############ main program starts here  ######################
turtle.tracer(0,0)           # diasble auto-update so board built faster
maze = Maze()                # enable the maze class
sprite = sprite()            # enable the sprite  class
end = End()                  # enable End position class
walls =[]                    # create walls coordinate list
finish = []                  # enable the finish array

setupMaze(grid)              # call the setup maze function
turtle.update()              # update board
turtle.tracer(1,1)           # turn auto-update back on

while True:
        sprite.spriteRight()
        sprite.spriteDown()
        sprite.spriteleft()
        sprite.spriteUp()

        time.sleep(0.02)
