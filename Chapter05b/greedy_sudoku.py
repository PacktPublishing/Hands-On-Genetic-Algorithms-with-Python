from copy import deepcopy
import numpy as np
from itertools import chain
import functools
import operator
import math

SUDOKU_PUZZLE = [
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

problem = np.array(SUDOKU_PUZZLE)

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

print(possibilities)
print(solution)

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
                    print(f'found fixed solution for {row}, {col}')

    # Find solution from mutual exclusion of solution possibilities
    for row in range(9):
        for col in range(9):
            possible = possibilities[row][col]
            if len(possible) > 1:
                red_possibilities = deepcopy(possibilities)
                red_possibilities[row][col] = set()

                square_row = row // 3
                square_col = col // 3

                overlap1 = possible - set(chain(*red_possibilities[row]))
                overlap2 = possible - set(chain(*[el[col] for el in red_possibilities]))
                overlap3 = possible - set(chain(*[el[i]
                                                  for el in red_possibilities[square_row * 3: square_row * 3 + 3]
                                                  for i in range(square_col * 3, square_col * 3 + 3)]))

                for overlap in [overlap1, overlap2, overlap3]:
                    if len(overlap) == 1:
                        print(f'found range solution for {row}, {col}: ', overlap)
                        possibilities[row][col] = overlap
                        solution[row, col] = min(overlap)
                        progress = True
                        break

print(possibilities)
print(solution)

total = 1
reihenfolge = 1
for row in possibilities:
    non_fixed = 0
    for col in row:
        total *= len(col)
        non_fixed += len(col)>1

    reihenfolge *= math.factorial(non_fixed)


print('combinatorial: ', total)
print('reihenfolge: ', reihenfolge)
print(total / reihenfolge)

print(functools.reduce(operator.mul, [math.factorial(el) for el in np.sum((problem==0), axis=1)])/reihenfolge)
