import numpy as np
import turtle
from itertools import product
import random
import time
import copy
from joblib import Parallel, delayed


class RoundTripProblem:
    """This class encapsulates the Round Trip Problem as can be
    seen here: https://www.janko.at/Raetsel/Rundreise/index.htm.
    The tiles of the playing field are represented as an array of (x, y) coordinates.
    For GA purposes these are mapped to integers that can be treated as an ordered list.
    The total distance can be calculated for a path represented by this list of tile indices.
    A turtle plot can be created for a path represented by a list of tile indices.

    :order: The size of the playing field (one side in squares).
    :holes: Array of (x, y) coordinates to be blacked out on playing field.
    """

    def __init__(self, order, holes):
        """
        Creates an instance of a RTRP

        :order: The size of the playing field (one side in squares).
        :holes: Array of (x, y) coordinates to be blacked out on playing field.
        """

        # initialize instance variables:
        self.order = order
        self.holes = holes

        # build coordinate list and remove holes
        self.tiles = [coord for coord in product(range(self.order), repeat=2) if coord not in self.holes]
        self.orient_dict = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        self.possible_dirs = self.getPossibleDirs(self.tiles)

    def __len__(self):
        """
        returns the number of indices used to internally represent the RTRP
        :return: the number of indices used to internally represent the RTRP
        """
        return len(self.tiles)

    def getPossibleDirs(self, tiles):
        """For a list of tiles generate a list of lists with potential outgoing arrows from each tile.

        :param tiles: list of x, y tuples.
        :return: list of lists of integers between 0 and 3 with same length as tiles.
        """

        dirs_list = list()
        for i in range(len(tiles)):
            dirs = list()
            for j in range(4):
                tile = tiles[i]
                offsets = self.orient_dict[j]
                tile_next = (tile[0] + offsets[0], tile[1] + offsets[1])
                if tile_next in tiles:
                    dirs.append(j)
            dirs_list.append(dirs)

        return dirs_list

    def get_sequence(self, start, tiles, directions):

        directions_dict = dict(zip(tiles, directions))
        to_walk = tiles.copy()
        path_walked = [start]
        dir_walked = []

        while True:
            right, up = self.orient_dict[directions_dict[start]]
            start = (start[0] + right, start[1] + up)
            if start in to_walk:
                to_walk.remove(start)
                path_walked.append(start)
                dir_walked.append((right, up))
            else:
                break

        return len(path_walked), path_walked

    def get_longest_sequence(self, directions):
        longest = 0
        longest_path = self.tiles[0]
        for tile in self.tiles:
            length, path = self.get_sequence(tile, self.tiles, directions)
            if length > longest:
                longest = length
                longest_path = path

        return longest, longest_path

    def plotData(self, directions):
        """plots the path described by the given directions starting from the tiles.

        :param directions: A list of integers 0 to 3 describing the given direction starting from a tile.
        :return: nothing
        """

        bob = turtle.Turtle()
        bob.color('black', 'red')

        def draw_lines(ind, pos0, orient, step, order):
            pos_xn, pos_yn = pos0
            angle = 0 if orient == 'h' else 270
            ind.setheading(angle)
            for i in range(order - 1):
                ind.penup()
                if orient == 'h':
                    pos_yn += -step
                else:
                    pos_xn += step
                ind.setposition(pos_xn, pos_yn)
                bob.pendown()
                bob.forward(step * order)

        def draw_square(ind, pos0, size, fill=False):
            speed = ind.speed()
            ind.speed(10)
            pos_xn, pos_yn = pos0
            ind.penup()
            ind.setheading(0)
            ind.setposition(pos_xn, pos_yn)
            ind.pendown()
            if fill:
                ind.begin_fill()
            for i in range(4):
                ind.forward(size)
                ind.right(90)
            if fill:
                ind.end_fill()
            ind.speed(speed)

        def draw_grid(ind, order, step):

            speed = ind.speed()
            ind.speed(10)
            pos_x, pos_y = -300, 300

            draw_lines(ind, (pos_x, pos_y), 'h', step, order)
            draw_lines(ind, (pos_x, pos_y), 'v', step, order)
            draw_square(ind, (pos_x, pos_y), step * order)
            ind.speed(speed)

        # draw the basic playing field
        step0 = int(600 / self.order)
        draw_grid(bob, self.order, step0)

        # draw holes (i.e. tiles not to be used)
        for hole in self.holes:
            hole_x = hole[0] * step0 - 300
            hole_y = -hole[1] * step0 + 300
            draw_square(bob, (hole_x, hole_y), step0, True)

        # draw solution path
        _, path = self.get_longest_sequence(directions)

        start_x = path[0][0] * step0 - 300 + 0.5 * step0
        start_y = -path[0][1] * step0 + 300 - 0.5 * step0

        bob.penup()
        bob.setposition(start_x, start_y)
        bob.pendown()
        bob.pencolor('red')

        for waypoint in path[1:]:
            new_x = waypoint[0] * step0 - 300 + 0.5 * step0
            new_y = -waypoint[1] * step0 + 300 - 0.5 * step0
            bob.setposition(new_x, new_y)

        turtle.done()


# testing the class:
def main():
    # create a problem instance:
    rtrp = RoundTripProblem(8, [(1, 1), (0, 4), (2, 3), (2, 5), (3, 2), (4, 7), (6, 2), (6, 6)])

    # generate a random solution and evaluate it:
    random.seed(42)
    randomSolution = random.choices(range(0, 4), k=len(rtrp))
    print(randomSolution)
    print(rtrp.get_longest_sequence(randomSolution))

    # see http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/bayg29.opt.tour
    # optimalSolution = [0, 27, 5, 11, 8, 25, 2, 28, 4, 20, 1, 19, 9, 3, 14, 17, 13, 16, 21, 10, 18, 24, 6, 22, 7, 26, 15, 12, 23]

    # print("Problem tiles: ", rtrp.tiles)
    # # print("Optimal solution = ", optimalSolution)
    # # print("Optimal distance = ", tsp.getTotalDistance(optimalSolution))
    #
    # # plot the solution:
    rtrp.plotData(randomSolution)
    # print(rtrp.getTotalDistance(randomSolution))


if __name__ == "__main__":
    main()
