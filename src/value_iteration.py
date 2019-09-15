#!/usr/bin/env python3

'''
This class implements a Value Iteration model used for reinforcement learning.
The model trains a race car to learn to get from the starting line to the
finish line in a minimum amount of time
'''

from racetrack import Racetrack
from agent import Agent
from helper import Helper
import numpy as np
import random
import copy

class ValueIteration(Helper):
    def __init__(self, file):
        racetrack = Racetrack(file)
        Helper.__init__(self, racetrack)
        self.accelerations = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], \
                              [1,-1], [1,0], [1,1]] # List of possible actions
        self.iterations = 0 # Number of training iterations
        self.past_value_difference = [] # Value difference of each iteration
        self.q_table = self.initialize_q(len(self.track.track),
                                         len(self.track.track[0]))
        self.v_table = self.initialize_vp(len(self.track.track),
                                          len(self.track.track[0]))
        self.p_table = self.initialize_vp(len(self.track.track),
                                          len(self.track.track[0]))

    def train(self, iterations, crash_type):
        '''
        This method trains the model

        INPUT:
            iterations(int): Maximum number of training iterations allowed
            crash_type(int): 0 for mild crash and 1 for harsh crash
        OUTPUT:
            list: Maximum value difference between each pair of iterations
        '''
        reward = -1
        threshold = 0.1
        discount = 0.95
        max_delta_q = 0
        car = self.initialize_car(crash_type)

        # Keep training until the maximum value difference between 2 iterations
        # is less than the threshold, or maximum iterations are used
        is_converged = False
        while (not is_converged) and (self.iterations < iterations):
            old_v_table = copy.deepcopy(self.v_table)
            max_delta_q = 0

            # Update the value table and the policy table for each state
            for i in range(len(self.v_table)):
                y_position = self.v_table[i]
                for j in range(len(y_position)):
                    x_velocity = self.v_table[i][j]
                    for k in range(len(x_velocity)):
                        y_velocity = self.v_table[i][j][k]
                        for l in range(len(y_velocity)):
                            policy = [0, 0]
                            max_q = -100000

                            # Update the Q table for each state-action pair
                            for m in range(len(self.accelerations)):
                                acceleration = self.accelerations[m]
                                car.position = [i, j]
                                car.velocity = [k - 5, l - 5]

                                # Update position after updating velocity
                                car.update_velocity(acceleration)
                                has_finished = car.update_position()
                                new_x_velocity = car.velocity[0] + 5
                                new_y_velocity = car.velocity[1] + 5
                                new_x_position = car.position[0]
                                new_y_position = car.position[1]

                                # Calculate the value of the next state
                                new_v = 0
                                reward = -1
                                if has_finished:
                                    reward = 0
                                else:
                                    new_v = old_v_table[new_x_position] \
                                                       [new_y_position] \
                                                       [new_x_velocity] \
                                                       [new_y_velocity]

                                # Calculate Q value of the state-action pair
                                new_q = reward + discount*new_v
                                self.q_table[i][j][k][l][m] = new_q

                                if new_q > max_q:
                                    policy = acceleration
                                    max_q = new_q

                            # Based on the best action, update correspondingly
                            old_q = self.v_table[i][j][k][l]
                            self.v_table[i][j][k][l] = max_q
                            self.p_table[i][j][k][l] = policy

                            # Calculate the maximum value difference
                            delta_q = old_q - max_q
                            if delta_q > max_delta_q:
                                max_delta_q = delta_q

            # For demonstration purpose
            print('Iteration:', self.iterations + 1)
            print('Maximum value difference:', max_delta_q)

            # Convergence criteria
            if max_delta_q < threshold:
                is_converged = True

            self.past_value_difference.append(max_delta_q)
            self.iterations += 1

        return self.past_value_difference

    def test(self, crash_type, write_to_file=False):
        '''
        This method simulates a race of a test car using the trained model

        INPUT:
            crash_type(int): 0 for mild crash and 1 for harsh crash
            write_to_file(boolean): True if the process is printed to console,
                                    False if the process is written to file
        OUTPUT:
            int: Number of steps needed for the test car to reach the goal
        '''
        car = self.initialize_car(crash_type)
        self.iter = 0
        self.record.append(car.position)

        # Allow the test car to run on the racetrack until it reaches the goal
        finish = False
        while not finish:
            # For demonstration purpose
            if not write_to_file:
                self.print_one(car)

            # Find the policy for current state
            i = car.position[0]
            j = car.position[1]
            k = car.velocity[0] + 5
            l = car.velocity[1] + 5
            acceleration = self.p_table[i][j][k][l]

            # Accelerate and move to the next state
            car.update_velocity(acceleration)
            finish = self.act(car)

        # For demonstration purpose
        if not write_to_file:
            self.print_last(car)
            print()
            self.print_all()

        return self.iter

    def initialize_q(self, x_size, y_size):
        '''
        This method initializes the Q table

        INPUT:
            x_size(int): The width of the racetrack
            y_size(int): The height of the racetrack
        '''
        q_table = []
        for i in range(x_size):
            x_position = []
            for j in range(y_size):
                y_position = []
                for k in range(-5, 6):
                    x_velocity = []
                    for l in range(-5, 6):
                        y_velocity = []
                        for m in range(len(self.accelerations)):
                            y_velocity.append(0)
                        x_velocity.append(y_velocity)
                    y_position.append(x_velocity)
                x_position.append(y_position)
            q_table.append(x_position)

        return q_table

    def initialize_vp(self, x_size, y_size):
        '''
        This method initializes the value table and the policy table

        INPUT:
            x_size(int): The width of the racetrack
            y_size(int): The height of the racetrack
        '''
        vp_table = []
        for i in range(x_size):
            x_position = []
            for j in range(y_size):
                y_position = []
                for k in range(11):
                    x_velocity = []
                    for l in range(11):
                        x_velocity.append(0)
                    y_position.append(x_velocity)
                x_position.append(y_position)
            vp_table.append(x_position)

        return vp_table

def main():
    print()
    file = input('Specify the racetrack to be used: ')
    iterations = 1000

    for crash_type in ([0, 1]):
        if crash_type == 0:
            crash = 'Mild Crash'
        else:
            crash = 'Harsh Crash'
        print()
        _ = input('Press enter to start training a Value Iteration model with '
                  + crash + ': ')
        print()
        print('Training process')
        print('----------------')

        value_iter = ValueIteration(file)
        learning_curve = value_iter.train(iterations, crash_type)

        print()
        print('--> Training completed')
        print()
        _ = input('Press enter to start testing: ')

        steps = value_iter.test(crash_type)
        steps += 1

        print('---------------------------------------------------------')
        print('Steps needed for the test car to reach the finish line:', steps)

    print()

if __name__ == '__main__':
	main()
