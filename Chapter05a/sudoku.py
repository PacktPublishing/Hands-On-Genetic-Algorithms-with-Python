import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


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

    # def __len__(self):
    #     """
    #     :return: the number of queens
    #     """
    #     return self.numOfQueens

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
            if solution[position] != self.preset[position]:
                violations += 1

        return violations

    def get_position_violation_count(self, solution):
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

        # vertical violations:
        for i in range(9):
            if len(set(solution[:, i])) < 9:
                violations += 1

        # horizontal violations:
        for i in range(9):
            if len(set(solution[i, :])) < 9:
                violations += 1

        # sector violations:
        for i in range(0, 8, 3):
            for j in range(0, 8, 3):
                if len(set(np.reshape(solution[i:i+3, j:j+3], -1, 1))) < 9:
                    violations += 1

        return violations

    def plot_sudoku(self, positions):
        """
        Plots the positions of the queens on the board according to the given solution
        :param positions: a list of indices corresponding to the positions of the queens in each row.
        """

        print(self.sudoku)

        # if len(positions) != self.numOfQueens:
        #     raise ValueError("size of positions list should be equal to ", self.numOfQueens)
        #
        # fig, ax = plt.subplots()
        #
        # # start with the board's squares:
        # board = np.zeros((self.numOfQueens, self.numOfQueens))
        # # change color of every other square:
        # board[::2, 1::2] = 1
        # board[1::2, ::2] = 1
        #
        # # draw the squares with two different colors:
        # ax.imshow(board, interpolation='none', cmap=mpl.colors.ListedColormap(['#ffc794', '#4c2f27']))
        #
        # # read the queen image thumbnail and give it a spread of 70% of the square dimensions:
        # queenThumbnail = plt.imread('queen-thumbnail.png')
        # thumbnailSpread = 0.70 * np.array([-1, 1, -1, 1]) / 2  # spread is [left, right, bottom, top]
        #
        # # iterate over the queen positions - i is the row, j is the column:
        # for i, j in enumerate(positions):
        #     # place the thumbnail on the matching square:
        #     ax.imshow(queenThumbnail, extent=[j, j, i, i] + thumbnailSpread)
        #
        # # show the row and column indexes:
        # ax.set(xticks=list(range(self.numOfQueens)), yticks=list(range(self.numOfQueens)))
        #
        # ax.axis('image')   # scale the plot as square-shaped
        #
        # return plt


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

    new_sudoku.plot_sudoku(1)

    violation_test = np.array(
        [
            [1, 0, 0, 6, 7, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 5, 7, 0],
            [6, 0, 0, 0, 9, 0, 0, 0, 7, 8],
            [0, 0, 0, 0, 0, 0, 3, 0, 4, 1],
            [0, 0, 0, 0, 6, 0, 3, 0, 0, 0],
            [7, 2, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 0, 0, 2, 0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 9, 0, 3],
            [3, 5, 2, 0, 0, 0, 0, 9, 0, 0]
        ]
    )

    # should return 2
    print('Preset Violations :', new_sudoku.get_preset_violation_count(violation_test))

    # should return 27
    print('Position Violations :', new_sudoku.get_position_violation_count(violation_test))


if __name__ == "__main__":
    main()

