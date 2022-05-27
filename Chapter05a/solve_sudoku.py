from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import elitism
import sudoku

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

# Genetic Algorithm constants:
POPULATION_SIZE = 2000
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 10
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual

# set the random seed for repeatable results
RANDOM_SEED = 50
random.seed(RANDOM_SEED)

# create the desired sudoku problem
n_sudoku = sudoku.SudokuProblem(SUDOKU_PUZZLE)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list of lists:
creator.create("Individual", np.ndarray, typecode='i', fitness=creator.FitnessMin)


def random_sudoku(size=9):
    return np.array([list(random.sample(range(size), size)) for _ in range(size)])


# create an operator that generates randomly shuffled indices:
toolbox.register("randomSudoku", random_sudoku, len(SUDOKU_PUZZLE))

# create the individual creation operator to fill up an Individual instance with shuffled indices:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomSudoku)

# create the population creation operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation - compute the total distance of the list of cities represented by indices:
def get_violations_count(individual):
    violations = n_sudoku.get_preset_violation_count(individual) + n_sudoku.get_position_violation_count(individual)
    return violations,  # return a tuple


toolbox.register("evaluate", get_violations_count)


# Genetic operators:

def multiline_upmx(ind1, ind2, indpb):
    pairs = list(zip(ind1, ind2))
    for i in range(len(pairs)):
        ind_i1, ind_i2 = tools.cxUniformPartialyMatched(pairs[i][0], pairs[i][1], indpb)
        ind1[i] = ind_i1
        ind2[i] = ind_i2

    return ind1, ind2


def multiline_shuffle_ind(ind, indpb):
    for i in range(len(ind)):
        ind_i = tools.mutShuffleIndexes(ind[i], indpb)
        ind[i] = ind_i[0]

    return ind,


def np_equal(a, b):
    return np.all(a==b)


toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", multiline_upmx, indpb=2.0/len(SUDOKU_PUZZLE))
toolbox.register("mutate", multiline_shuffle_ind, indpb=1.0/len(SUDOKU_PUZZLE))


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    new_population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE, similar=np_equal)

    # perform the Genetic Algorithm flow with hof feature added:
    # new_population, logbook = elitism.eaSimpleWithElitism(new_population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
    #                                                       ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    new_population, logbook = algorithms.eaSimple(new_population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                          ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,
                                                          verbose=True)

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
    sns.set_style("whitegrid", {'axes.grid' : False})
    n_sudoku.plot_sudoku(hof.items[0])

    # show both plots:
    plt.show()


if __name__ == "__main__":
    main()
