#!/usr/bin/env python3

'''
This class stores the information about a racetrack on which the race car runs
'''

import numpy as np

class Racetrack():
    def __init__(self, file):
        self.file = file # Name of the racetrack file

        # Stores the racetrack in a 2D list and remove the first line
        self.track = []
        for line in open(file):
            line = line.rstrip()
            lst = list(line)
            self.track.append(lst)
        self.track = self.track[1:]

        self.start_locations = []  # Stores all start points
        self.finish_locations = [] # Stores all finish points
        self.track_locations = []  # Stores all track points
        self.wall_locations = []   # Stores all wall points
        for i, row in enumerate(self.track):
            for j, val in enumerate(row):
                if val == 'S':
                    self.start_locations.append((i, j))
                elif val == 'F':
                    self.finish_locations.append((i, j))
                elif val == '.':
                    self.track_locations.append((i, j))
                elif val == '#':
                    self.wall_locations.append((i, j))

        self.finish_line = self.draw_finish_line() # Position of finish line

    def get_nearest_track(self, position):
        '''
        This method finds a nearest track point to a point in the racetrack

        INPUT:
            position(tuple): Coordinates of a specific point
        OUTPUT:
            tuple: Coordinates of the nearest track point
        '''
        position = np.array(position)
        track_locations = np.array(self.track_locations)
        distance = np.sum(np.square(track_locations - position), axis=1)
        nearest = np.argmin(distance)
        nearest_track = self.track_locations[nearest]

        return nearest_track

    def get_nearest_start(self, position):
        '''
        This method finds a nearest start point to a point in the racetrack

        INPUT:
            position(tuple): Coordinates of a specific point
        OUTPUT:
            tuple: Coordinates of the nearest start point
        '''
        position = np.array(position)
        start_locations = np.array(self.start_locations)
        distance = np.sum(np.square(start_locations - position), axis=1)
        nearest = np.argmin(distance)
        nearest_start = self.start_locations[nearest]

        return nearest_start

    def is_wall(self, position):
        '''
        This method determines whether a point is a wall point or not

        INPUT:
            position(tuple): Coordinates of a specific point
        OUTPUT:
            boolean: True if the point is a wall point, False otherwise
        '''
        if (position[0] > (len(self.track) - 1) or
            position[1] > (len(self.track[0]) - 1)):
            return True
        elif self.track[position[0]][position[1]] == '#':
            return True
        elif position[0] < 0 or position[1] < 0:
            return True

        return False

    def draw_finish_line(self):
        '''
        Find the direction and location of the finish line

        OUTPUT:
            boolean: True if the finish line is vertical, False if horizontal
            tuple: Coordinates of the point on one end of the finish line
            tuple: Coordinates of the point on the other end of the finish line
        '''
        max_distance = 0
        best_end1 = None # Point on one end of the finish line
        best_end2 = None # Point on the other end of the finish line
        for i in range(len(self.finish_locations)):
            end1 = np.array(self.finish_locations[i])
            for j in range(len(self.finish_locations)):
                end2 = np.array(self.finish_locations[j])
                distance = np.sum(np.abs(end1 - end2))
                if distance > max_distance:
                    best_end1 = end1
                    best_end2 = end2
                    max_distance = distance

        is_vertical = True
        if (best_end1 - best_end2)[0] == 0:
            is_vertical = False

        return (is_vertical, best_end1, best_end2)

    def check_finish_line(self, position1, position2):
        '''
        This method checks if the car has passed the finish line

        INPUT:
            position1(tuple): Coordinates of the car's current position
            position2(tuple): Coordinates of the car's next position
        OUTPUT:
            boolean: True if the car has passed the finish line
        '''
        across_line = False
        within_ends = False
        crossed = False
        position1 = np.array(position1)
        position2 = np.array(position2)

        if self.finish_line[0]: # Finish line is vertical
            line = self.finish_line[1][1]
            if (line-position1[1]) * (line-position2[1]) <= 0:
                across_line = True
            end1 = self.finish_line[1][0]
            end2 = self.finish_line[2][0]
            if end1 <= position1[0] <= end2 or end2 <= position1[0] <= end1:
                within_ends = True
            if across_line and within_ends:
                crossed = True
        else: # Finish line is horizontal
            line = self.finish_line[1][0]
            if (line-position1[0]) * (line-position2[0]) <= 0:
                across_line = True
            end1 = self.finish_line[1][1]
            end2 = self.finish_line[2][1]
            if end1 <= position1[1] <= end2 or end2 <= position1[1] <= end1:
                within_ends = True
            if across_line and within_ends:
                crossed = True

        return crossed
