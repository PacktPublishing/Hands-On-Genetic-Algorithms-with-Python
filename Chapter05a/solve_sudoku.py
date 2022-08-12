from deap import base
from deap import creator
from deap import tools

import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import elitism
import sudoku
import greedy_sudoku


def random_sudoku(size, lengths):
    new_sudoku = list()
    for i in range(size):
        new_row = random.sample(range(lengths[i]), lengths[i])
        new_sudoku.append(new_row)

    return new_sudoku


# fitness calculation - compute the total distance of the list of cities represented by indices:
def get_violations_count(individual, sudoku_problem):
    violations = sudoku_problem.get_position_violation_count(individual)
    return violations,  # return a tuple


# Genetic operators:
def swap_cx(ind1, ind2, indpb):
    for i in range(len(ind1)):
        if random.random() <= indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def simple_swap_mutation(ind, indpb):
    for i in range(len(ind)):
        if random.random() <= indpb:
            id1, id2 = random.sample(range(len(ind[i])), 2)
            ind[i][id1], ind[i][id2] = ind[i][id2], ind[i][id1]

    return ind,


def np_equal(a, b):
    return np.all(a == b)


# Genetic algorithm main flow:
def ga_main(partial_solution):

    # Genetic Algorithm constants:
    POPULATION_SIZE = 10000
    MAX_GENERATIONS = 1000
    HALL_OF_FAME_SIZE = 500
    P_CROSSOVER = 0.9  # probability for crossover
    P_MUTATION = 0.2  # probability for mutating an individual

    # set the random seed for repeatable results
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create the desired sudoku problem
    n_sudoku = sudoku.SudokuProblem(partial_solution)

    toolbox = base.Toolbox()

    # define a single objective, minimizing fitness strategy:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # TODO use itertools.product(*possibilities[i]) to generate an array of valid lines only
    # possibilities can be returned as additional result by the greedy function
    # the idea is to build individuals by sampling from valid rows
    # mating happens by swapping rows between two individuals
    # mutation picks from a list of valid options for a given row
    # this should massively reduce the number of solution candidates

    # create the Individual class based on list of lists:
    creator.create("Individual", list, typecode='i', fitness=creator.FitnessMin)

    # create an operator that generates randomly shuffled indices:
    toolbox.register("randomSudoku", random_sudoku, n_sudoku.size, n_sudoku.n_empty)

    # create the individual creation operator to fill up an Individual instance with shuffled indices:
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomSudoku)

    # create the population creation operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register("evaluate", get_violations_count, sudoku_problem=n_sudoku)

    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", swap_cx, indpb=1.0 / len(SUDOKU_PUZZLE))
    toolbox.register("mutate", simple_swap_mutation, indpb=1.0 / len(SUDOKU_PUZZLE))

    # create initial population (generation 0):
    new_population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE, similar=np_equal)

    # perform the Genetic Algorithm flow with hof feature added:
    new_population, logbook = elitism.eaSimpleWithElitism(new_population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                          ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,
                                                          verbose=True, stuck=(50, 'chernobyl'))

    # new_population, logbook = algorithms.eaSimple(new_population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
    #                                                       ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,
    #                                                       verbose=True)

    # print hall of fame members info:
    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])

    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # plot best solution:
    sns.set_style("whitegrid", {'axes.grid': False})
    n_sudoku.plot_solution(hof.items[0])

    # show both plots:
    plt.show()


# Hybrid solution flow:
def main(problem):
    # try greedy solution
    greedy_solution = greedy_sudoku.greedy_search(problem)
    print("Greedy solution is :\n", greedy_solution)
    empty = (greedy_solution == 0).sum()
    if empty == 0:
        print("Solution is :\n", greedy_solution)
    else:
        ga_main(greedy_solution)


if __name__ == "__main__":
    # problem constants:
    SUDOKU_PUZZLE = [
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

    main(SUDOKU_PUZZLE)
