#!/usr/bin/env python3

'''
This class implements a race car which runs on a racetrack implemented by the
"Racetrack" class
'''

from racetrack import Racetrack
import random

class Agent():
    def __init__(self, track, position, velocity, crash_type):
        self.track = track # A Racetrack object
        self.old_position = position # Coordinates of the car's last position
        self.position = position # Coordinates of the car's current position
        self.velocity = velocity # Velocity of the car

        # Specify one of the two types of crash
        if crash_type == 0:
            self.crash = self.mild_crash
        else:
            self.crash = self.harsh_crash

    def mild_crash(self):
        '''
        This method resets the car to the nearest track point when it crashes
        '''
        self.position = self.track.get_nearest_track(self.position)
        self.velocity = [0, 0]

    def harsh_crash(self):
        '''
        This method resets the car to the nearest start point when it crashes
        '''
        self.position = self.track.get_nearest_start(self.position)
        self.velocity = [0, 0]

    def update_position(self):
        '''
        This method updates the position of the car using its current velocity

        OUTPUT:
            boolean: True if the car will pass the finish line after the update
        '''
        self.old_position = self.position
        new_position = [self.position[0] + self.velocity[0], \
                        self.position[1] + self.velocity[1]]

        # 3 conditions: The car either passes the finish line, runs into a wall
        # or continues to run on the racetrack
        if self.track.check_finish_line(self.position, new_position):
            return True
        elif self.track.is_wall(new_position):
            self.crash()
        else:
            self.position = new_position

        return False

    def update_velocity(self, acceleration):
        '''
        This method updates the velocity of the car given acceleration vector

        INPUT:
            acceleration(list): y and x components of an acceleration vector
        '''
        x_velocity = self.velocity[0] + acceleration[0]
        y_velocity = self.velocity[1] + acceleration[1]

        # Velocity cannot exceed the [-5, 5] bound
        if abs(x_velocity) <= 5:
            self.velocity[0] = x_velocity
        if abs(y_velocity) <= 5:
            self.velocity[1] = y_velocity
