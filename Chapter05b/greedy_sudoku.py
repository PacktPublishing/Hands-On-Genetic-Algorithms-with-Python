from copy import deepcopy
import numpy as np
from itertools import chain

SOLVABLE = [
    [0, 0, 0,   0, 1, 3,   0, 0, 0],
    [0, 0, 0,   0, 0, 0,   6, 7, 4],
    [0, 4, 9,   0, 0, 0,   0, 0, 0],

    [0, 0, 0,   0, 0, 0,   5, 9, 0],
    [0, 0, 0,   0, 8, 7,   0, 0, 0],
    [6, 9, 1,   0, 0, 0,   0, 0, 0],

    [1, 0, 0,   7, 0, 4,   0, 0, 0],
    [2, 0, 0,   6, 0, 0,   0, 0, 1],
    [0, 5, 0,   0, 0, 0,   0, 4, 3]
]

NOT_SOLVABLE = [
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 0, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 0],
            [4, 0, 0, 8, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 7, 0]
        ]


def greedy_search(sudoku_problem):
    """
    Takes in a sudoku problem as a 9x9 array, where missing numbers are represented as zeroes.
    Returns a list of lists of the same format with missing numbers completed by greedy algorithm where possible.
    Does not but could return list of sets of possible solutions for non-solved fields.
    params:
        sudoku_problem (array): sudoku problem as nested 9x9 array
    returns:
        solution (list): (partially) completed solution as nested 9x9 list
    """

    if isinstance(sudoku_problem, np.ndarray):
        problem = sudoku_problem
    else:
        problem = np.array(sudoku_problem)

    solution = problem.copy()

    possibilities = list()
    for row in range(9):
        row_list = list()
        for col in range(9):
            if solution[row, col] > 0:
                row_list.append({solution[row, col]})
            else:
                row_list.append(set(range(1, 10)))
        possibilities.append(row_list)

    full_range = set(range(1, 10))

    progress = True

    while progress:
        progress = False

        # Find solution from mutual exclusion of fixed fields
        for row in range(9):
            for col in range(9):
                if solution[row, col] == 0:
                    square_row = row // 3
                    square_col = col // 3
                    possible_range = full_range - set(solution[row, :])
                    possible_range = possible_range - set(solution[:, col])
                    possible_range = possible_range - set(solution[
                                                          square_row * 3: square_row * 3 + 3,
                                                          square_col * 3: square_col * 3 + 3].flat)

                    if len(possible_range) < len(possibilities[row][col]):
                        progress = True

                    possibilities[row][col] = possible_range

                    if len(possible_range) == 1:
                        solution[row, col] = min(possible_range)

        # Find solution from mutual exclusion of solution possibilities
        for row in range(9):
            for col in range(9):
                possible = possibilities[row][col]
                if len(possible) > 1:
                    red_possibilities = deepcopy(possibilities)
                    red_possibilities[row][col] = set()

                    square_row = row // 3
                    square_col = col // 3

                    # this could be solved more elegantly, works perfectly though
                    overlap1 = possible - set(chain(*red_possibilities[row]))
                    overlap2 = possible - set(chain(*[el[col] for el in red_possibilities]))
                    overlap3 = possible - set(chain(*[el[i]
                                                      for el in red_possibilities[square_row * 3: square_row * 3 + 3]
                                                      for i in range(square_col * 3, square_col * 3 + 3)]))

                    for overlap in [overlap1, overlap2, overlap3]:
                        if len(overlap) == 1:
                            possibilities[row][col] = overlap
                            solution[row, col] = min(overlap)
                            progress = True
                            break

    return solution, possibilities


if __name__=='__main__':
    print(np.array(SOLVABLE))
    print(greedy_search(SOLVABLE))
    print(np.array(NOT_SOLVABLE))
    print(greedy_search(NOT_SOLVABLE))

