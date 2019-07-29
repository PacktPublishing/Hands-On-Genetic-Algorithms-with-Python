import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class NQueensProblem:
    """This class encapsulates the N-Queens problem
    """

    def __init__(self, numOfQueens):
        """
        :param numOfQueens: the number of queens in the problem
        """
        self.numOfQueens = numOfQueens

    def __len__(self):
        """
        :return: the number of queens
        """
        return self.numOfQueens

    def getViolationsCount(self, positions):
        """
        Calculates the number of violations in the given solution
        Since the input contains unique indices of columns for each row, no row or column violations are possible,
        Only the diagonal violations need to be counted.
        :param positions: a list of indices corresponding to the positions of the queens in each row
        :return: the calculated value
        """

        if len(positions) != self.numOfQueens:
            raise ValueError("size of positions list should be equal to ", self.numOfQueens)

        violations = 0

        # iterate over every pair of queens and find if they are on the same diagonal:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):

                # first queen in pair:
                column1 = i
                row1 = positions[i]

                # second queen in pair:
                column2 = j
                row2 = positions[j]

                # look for diagonal threat for th ecurrent pair:
                if abs(column1 - column2) == abs(row1 - row2):
                    violations += 1

        return violations

    def plotBoard(self, positions):
        """
        Plots the positions of the queens on the board according to the given solution
        :param positions: a list of indices corresponding to the positions of the queens in each row.
        """

        if len(positions) != self.numOfQueens:
            raise ValueError("size of positions list should be equal to ", self.numOfQueens)

        fig, ax = plt.subplots()

        # start with the board's squares:
        board = np.zeros((self.numOfQueens, self.numOfQueens))
        # change color of every other square:
        board[::2, 1::2] = 1
        board[1::2, ::2] = 1

        # draw the squares with two different colors:
        ax.imshow(board, interpolation='none', cmap=mpl.colors.ListedColormap(['#ffc794', '#4c2f27']))

        # read the queen image thumbnail and give it a spread of 70% of the square dimensions:
        queenThumbnail = plt.imread('queen-thumbnail.png')
        thumbnailSpread = 0.70 * np.array([-1, 1, -1, 1]) / 2  # spread is [left, right, bottom, top]

        # iterate over the queen positions - i is the row, j is the column:
        for i, j in enumerate(positions):
            # place the thumbnail on the matching square:
            ax.imshow(queenThumbnail, extent=[j, j, i, i] + thumbnailSpread)

        # show the row and column indexes:
        ax.set(xticks=list(range(self.numOfQueens)), yticks=list(range(self.numOfQueens)))

        ax.axis('image')   # scale the plot as square-shaped

        return plt


# testing the class:
def main():
    # create a problem instance:
    nQueens = NQueensProblem(8)

    # a known good solution:
    #solution = [5, 0, 4, 1, 7, 2, 6, 3]

    # a solution with 3 violations:
    solution = [1, 2, 7, 5, 0, 3, 4, 6]

    print("Number of violations = ", nQueens.getViolationsCount(solution))

    plot = nQueens.plotBoard(solution)
    plot.show()


if __name__ == "__main__":
    main()

