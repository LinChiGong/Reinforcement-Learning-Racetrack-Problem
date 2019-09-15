#!/usr/bin/env python3

'''
This is an auxiliary class for both the "ValueIteration" class and the
"QLearning" class
'''

from racetrack import Racetrack
from agent import Agent
import numpy as np
import random
import copy

class Helper:
    def __init__(self, track):
        self.track = track # A Racetrack object
        self.print_track = copy.deepcopy(track.track) # Copy of the racetrack
        self.cache = 'S' # Memorize the type of point the car was located at
        self.record = [] # Stores all the past locations the car was at
        self.iter = 0 # Number of iterations for the test car to reach the goal
        self.print_start = True # Indicates the first iteration

    def initialize_car(self, crash_type):
        '''
        This method initializes a car at a random start point

        INPUT:
            crash_type(int): 0 for mild crash and 1 for harsh crash
        OUTPUT:
            Agent: The initial car as an "Agent" object
        '''
        start = random.choice(self.track.start_locations)
        start_velocity = [0, 0]
        car = Agent(self.track, start, start_velocity, crash_type)

        return car

    def act(self, agent):
        '''
        This method updates the position of the test car

        INPUT:
            agent(Agent): The test car
        OUTPUT:
            boolearn: True if the test car has reached the finish line
        '''
        # Check if the car has reached the finish line
        has_finished = agent.update_position()
        if has_finished:
            print()
            print('--> Last acceleration')
            return True

        self.iter += 1
        self.record.append(agent.position)

        # Update the position of the test car in the racetrack to be printed
        # Test car is represented as "C" when printed in the racetrack
        if agent.position != agent.old_position:
            self.print_track[agent.old_position[0]][agent.old_position[1]] \
                                                                   = self.cache
            self.cache = self.print_track[agent.position[0]][agent.position[1]]
            self.print_track[agent.position[0]][agent.position[1]] = 'C'

        return False

    def print_one(self, agent):
        '''
        This method prints the state of the test car in each iteration

        INPUT:
            agent(Agent): The test car
        '''
        print()
        print('Iteration:', self.iter)
        print('Velocity:', agent.velocity)
        print('------------')

        if self.print_start:
            self.print_track[agent.old_position[0]][agent.old_position[1]] \
                                                                   = self.cache
            self.cache = self.print_track[agent.position[0]][agent.position[1]]
            self.print_track[agent.position[0]][agent.position[1]] = 'C'
            self.print_start = False

        for row in self.print_track:
            for val in row:
                print(val, end='')
            print()

    def print_last(self, agent):
        '''
        This method prints the state of the test car in the last iteration

        INPUT:
            agent(Agent): The test car
        '''
        print()
        print('Velocity:', agent.velocity)
        print('------------')
        for row in self.print_track:
            for val in row:
                print(val, end='')
            print()
        print()
        print('--> Finished!!')

    def print_all(self):
        '''
        This method prints the states of the test car in all iterations
        '''
        print('Car positions from all iterations')
        print('---------------------------------')

        for i in range(len(self.record)):
            position = self.record[i]
            self.print_track[position[0]][position[1]] = i

        for row in self.print_track:
            for val in row:
                print(val, end='')
            print()

    def write_all(self, out_file):
        '''
        This method writes the states of the test car in all iterations to an
        output file

        INPUT:
            out_file(_io.TextIOWrapper): The output file being written to
        '''
        out_file.write('Car positions from all iterations\n')
        out_file.write('---------------------------------\n')

        temp_track = copy.deepcopy(self.track.track)
        for i in range(len(self.record)):
            position = self.record[i]
            temp_track[position[0]][position[1]] = i

        for row in temp_track:
            for val in row:
                out_file.write(str(val))
            out_file.write('\n')

        self.record.clear()
