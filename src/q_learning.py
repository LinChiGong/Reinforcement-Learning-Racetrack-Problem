#!/usr/bin/env python3

'''
This class implements a Q-Learning model used for reinforcement learning.
The model trains a race car to learn to get from the starting line to the
finish line in a minimum amount of time
'''

from racetrack import Racetrack
from agent import Agent
from helper import Helper
import numpy as np
import random
import copy

class QLearning(Helper):
    def __init__(self, file):
        racetrack = Racetrack(file)
        Helper.__init__(self, racetrack)
        self.q_table = self.initialize_q(len(self.track.track),
                                         len(self.track.track[0]))
        self.num_steps = [] # Number of steps taken in each training iteration

    def train(self, iterations, crash_type, steps):
        '''
        This method trains the model

        INPUT:
            iterations(int): Number of training iterations
            crash_type(int): 0 for mild crash and 1 for harsh crash
            steps(int): Maximum number of training steps in each iteration
        OUTPUT:
            list: Numbers of steps needed to reach the goal in all iterations
        '''
        reward = -1
        discount = 0.95
        max_delta_q = 0
        epsilon = 0.5
        learning_rate = 0.75
        decay = 0.9999

        # Train through the specified number of training iterations
        for iteration in range(iterations):
            car = self.initialize_car(crash_type)

            # [i][j][k][l] is the state of the car
            i = car.position[0]
            j = car.position[1]
            k = car.velocity[0] + 5
            l = car.velocity[1] + 5

            # Keep training until reaching the finish line, or the maximum
            # number of steps are taken
            step = 0
            finish = False
            while (not finish) and (step < steps):
                q_values = self.q_table[i][j][k][l]
                acceleration_index = self.epsilon_greedy_policy(q_values,
                                                                epsilon)

                # 20% of the time, the attempt to accelerate would fail
                q_value = q_values[1][1]
                acceleration = [0, 0]
                if random.random() < 0.8:
                    q_value = q_values[acceleration_index[0]] \
                                      [acceleration_index[1]]
                    acceleration = [acceleration_index[0] - 1, \
                                    acceleration_index[1] - 1]

                # Update position and velocity. Check if the goal is reached
                car.update_velocity(acceleration)
                has_finished = car.update_position()
                if has_finished:
                    finish = True
                else:
                    i = car.position[0]
                    j = car.position[1]
                    k = car.velocity[0] + 5
                    l = car.velocity[1] + 5

                    # Update the Q table for the current state-action pair
                    q_values_prime = self.q_table[i][j][k][l]
                    max_q_value_prime = np.max(q_values_prime)
                    q_values[acceleration_index[0]][acceleration_index[1]] += (
                        learning_rate * (reward + discount * max_q_value_prime
                        - q_value))

                step += 1

            # Gradually reduce epsilon and learning rate through the process
            epsilon *= decay
            if learning_rate > 0.01:
                learning_rate *= decay

            self.num_steps.append(step)

        return self.num_steps

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
        epsilon = 0 # Only do exploitation during testing

        # Allow the test car to run on the racetrack until it reaches the goal,
        # or it has taken 1000 steps
        finish = False
        while not finish and self.iter < 1000:
            # For demonstration purpose
            if not write_to_file:
                self.print_one(car)

            # Find the policy for current state
            i = car.position[0]
            j = car.position[1]
            k = car.velocity[0] + 5
            l = car.velocity[1] + 5
            q_values = self.q_table[i][j][k][l]
            acceleration_index = self.epsilon_greedy_policy(q_values, epsilon)
            acceleration = [acceleration_index[0]-1, acceleration_index[1]-1]

            # Accelerate and move to the next state
            car.update_velocity(acceleration)
            finish = self.act(car)

        # For demonstration purpose
        if not write_to_file:
            self.print_last(car)
            print()
            self.print_all()

        return self.iter

    def epsilon_greedy_policy(self, q_values, epsilon):
        '''
        This method uses the epsilon-greedy policy to choose an action

        INPUT:
            q_values(list): Q values of all possible actions
            epsilon(float): Probability of doing exploration
        OUTPUT:
            int: Index of the chosen action
        '''
        if np.random.random() < epsilon:
            # Explore
            acceleration_index = (np.random.randint(3), np.random.randint(3))
        else:
            # Exploit
            acceleration_index = np.unravel_index(q_values.argmax(),
                                                  q_values.shape)

        return acceleration_index

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
                        default = np.random.rand(3, 3)
                        default *= -1
                        x_velocity.append(default)
                    y_position.append(x_velocity)
                x_position.append(y_position)
            q_table.append(x_position)

        return q_table

def main():
    print()
    file = input('Specify the racetrack to be used: ')
    iterations = 1000
    max_steps = 1000

    for crash_type in ([0, 1]):
        if crash_type == 0:
            crash = 'Mild Crash'
        else:
            crash = 'Harsh Crash'
        print()
        _ = input('Press enter to start training a Q-Learning model with '
                  + crash + ': ')
        print()
        print('Training process')
        print('----------------')

        q_learning = QLearning(file)
        learning_curve = q_learning.train(iterations, crash_type, max_steps)

        for i in range(len(learning_curve)):
            print('Iteration:', i + 1)
            print('Steps taken:', learning_curve[i])
        print()
        print('--> Training completed')
        print()
        _ = input('Press enter to start testing: ')

        steps = q_learning.test(crash_type)
        steps += 1

        print('---------------------------------------------------------')
        print('Steps needed for the test car to reach the finish line:', steps)

    print()

if __name__ == '__main__':
	main()
