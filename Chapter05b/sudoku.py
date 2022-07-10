import numpy as np


class SudokuProblem:
    """This class encapsulates the Sudoku Problem
    """

    def __init__(self, sudoku):
        """
        :param sudoku: Array type of shape 9, 9 containing problem to solve. Empty cells are denoted as zero.
        """
        self.sudoku = np.array(sudoku)
        self.size = self.sudoku.shape[0]
        self.n_empty = [np.sum(self.sudoku[i, :] == 0) for i in range(self.size)]
        self.number_map = self.build_map()

    def build_map(self):
        """
        Builds a list that contains all numbers that have to be used to solve the sudoku problem.
        :return: list of numbers
        """
        number_map = list()
        for row in self.sudoku:
            preset = row[row.nonzero()]
            row_map = np.array([i for i in range(1, self.size + 1) if i not in preset])
            number_map.append(row_map)

        return number_map

    def map_solution(self, solution):
        """
        Takes an array of indices and maps the number map to the indices.
        Adds mapped numbers to empty sudoku fields and returns sudoku array of shape (x, x).
        :return: filled in solution.
        """
        mapped_solution = np.empty((self.size, self.size), int)
        for i in range(self.size):
            mapped_row = self.number_map[i][solution[i]]
            sudoku_row = self.sudoku[i].copy()
            sudoku_row[sudoku_row == 0] = mapped_row
            mapped_solution[i] = sudoku_row

        return mapped_solution

    def get_position_violation_count(self, solution):
        """
        Calculates the number of violations in the given solution.
        Since the input contains unique indices of columns for each row, no row or column violations are possible,
        Only the diagonal violations need to be counted.
        :param solution: Solution array type of shape 9, 9 containing a full solution.
        :return: the calculated value
        """

        # fill empty sudoku cells with solution
        mapped_solution = self.map_solution(solution)

        # count violations
        violations = 0

        # vertical violations:
        for i in range(9):
            violations += 9 - len(np.unique(mapped_solution[:, i]))

        # # horizontal violations:
        # for i in range(9):
        #     violations += 9 - len(set(mapped_solution[i, :]))

        # sector violations:
        for i in range(0, 8, 3):
            for j in range(0, 8, 3):
                violations += 9 - len(np.unique(mapped_solution[i:i+3, j:j+3]))

        # if 2 <= violations <= 4:
        #     violations = 2

        return violations

    def plot_solution(self, solution):
        """
        Plots a zero-based sudoku solution in the final one-based format
        :param solution: a sudoku solution (zero-based) to be printed
        """

        # fill empty sudoku cells with solution
        mapped_solution = self.map_solution(solution)

        print(mapped_solution)


# testing the class:
def main():
    # create a problem instance:
    new_sudoku = SudokuProblem(
        [
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
    )

    optimal_solution = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
    )

    eight_violations = np.array(
        [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 2, 7, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 9, 3],
            [4, 2, 6, 8, 5, 3, 7, 2, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 1, 6, 7, 9]
        ]
    )

    def make_solution(solution):

        solution_map = list()

        for i in range(new_sudoku.size):
            row_numbers = [el for el in solution[i] if el in new_sudoku.number_map[i]]
            row_indices = np.array([np.where(new_sudoku.number_map[i] == el) for el in row_numbers]).flatten()
            solution_map.append(row_indices)

        return solution_map

    print(new_sudoku.build_map())
    print(new_sudoku.size)
    print(new_sudoku.number_map)

    optimal = make_solution(optimal_solution)
    eight = make_solution(eight_violations)

    new_sudoku.plot_solution(optimal)
    print('Optimal solution check: ', new_sudoku.get_position_violation_count(optimal))
    new_sudoku.plot_solution(eight)
    print('Eight violations solution check: ', new_sudoku.get_position_violation_count(eight))


if __name__ == "__main__":
    main()

