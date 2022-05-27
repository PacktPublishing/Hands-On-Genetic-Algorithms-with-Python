import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math


class SudokuProblem:
    """This class encapsulates the Sudoku Problem
    """

    def __init__(self, sudoku):
        """
        :param sudoku: Array type of shape 9, 9 containing problem to solve. Empty cells are denoted as zero.
        """
        self.sudoku = np.array(sudoku)
        self.preset = dict(
            zip(
                list(zip(*np.nonzero(self.sudoku))),  # coordinates of nonzero input
                self.sudoku[np.nonzero(self.sudoku)]  # according input
            )
        )

    def get_preset_violation_count(self, solution):
        """
        Calculates the number of violations in the given solution
        Since the input contains unique indices of columns for each row, no row or column violations are possible,
        Only the diagonal violations need to be counted.
        :param solution: Solution array type of shape 9, 9 containing a full solution.
        :return: the calculated value
        """

        # add check: are all fields nonzero and lower 10
        # add check: is the solution an array of shape 9, 9

        violations = 0

        for position in self.preset:
            x, y = position
            if solution[x][y] + 1 != self.preset[position]:
                violations += 1

        return violations * 1

    def get_position_violation_count(self, solution):
        """
        Calculates the number of violations in the given solution.
        Since the input contains unique indices of columns for each row, no row or column violations are possible,
        Only the diagonal violations need to be counted.
        :param solution: Solution array type of shape 9, 9 containing a full solution.
        :return: the calculated value
        """

        # add check: are all fields nonzero and lower 10
        # add check: is the solution an array of shape 9, 9

        # add violations: as a sudoku object is defined as a collection of ordered lists (=row)
        # only vertical and sector (3x3 tile) violations have to be checked!
        violations = 0

        # vertical violations:
        for i in range(9):
            violations += 9 - len(set(solution[:, i]))

        # sector violations:
        for i in range(0, 8, 3):
            for j in range(0, 8, 3):
                violations += 9 - len(set(np.reshape(solution[i:i+3, j:j+3], -1, 1)))

        return violations

    def plot_sudoku(self, solution):
        """
        Plots a zero-based sudoku solution in the final one-based format
        :param solution: a sudoku solution (zero-based) to be printed
        """

        print(solution + 1)


# testing the class:
def main():
    # create a problem instance:
    new_sudoku = SudokuProblem(
        [
            [1, 0, 0, 5, 7, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 5, 7, 0],
            [6, 0, 0, 0, 9, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
            [0, 0, 0, 0, 6, 0, 3, 0, 0, 0],
            [7, 2, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 0, 2, 0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 2, 0, 3],
            [3, 5, 2, 0, 0, 0, 0, 9, 0, 0]
        ]
    )


            # a known good solution:
    #solution = [5, 0, 4, 1, 7, 2, 6, 3]

    # a solution with 3 violations:
    # solution = [1, 2, 7, 5, 0, 3, 4, 6]
    #
    # print("Number of violations = ", nQueens.getViolationsCount(solution))
    #
    # plot = nQueens.plotBoard(solution)
    # plot.show()

    violation_test = np.array(
        [
            [1, 0, 0, 6, 7, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 5, 7, 0],
            [6, 0, 0, 0, 8, 0, 0, 0, 7, 8],
            [0, 0, 0, 0, 0, 0, 3, 0, 4, 1],
            [0, 0, 0, 0, 6, 0, 3, 0, 0, 0],
            [7, 2, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 0, 0, 2, 0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 8, 0, 3],
            [3, 5, 2, 0, 0, 0, 0, 8, 0, 0]
        ]
    )

    new_sudoku.plot_sudoku(violation_test)

    # should return 2
    print('Preset Violations :', new_sudoku.get_preset_violation_count(violation_test))

    # should return 27
    print('Position Violations :', new_sudoku.get_position_violation_count(violation_test))


if __name__ == "__main__":
    main()

