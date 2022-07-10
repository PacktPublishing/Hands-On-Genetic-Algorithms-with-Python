from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import array
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import rtrp
import elitism_plus

import multiprocessing

# set the random seed for repeatable results
RANDOM_SEED = 100
random.seed(RANDOM_SEED)

# create the desired traveling salesman problem instace:
RTRP_ORDER = 8  # name of problem
RTRP_HOLES = [(1, 1), (0, 4), (2, 3), (2, 5), (3, 2), (4, 7), (6, 2), (6, 6)]
roundtrip = rtrp.RoundTripProblem(RTRP_ORDER, RTRP_HOLES)

# Genetic Algorithm constants:
POPULATION_SIZE = 50000
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 100
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.2   # probability for mutating an individual

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list of integers:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)


def restr_direction(dirs):
    return [random.sample(x, 1)[0] for x in dirs]


def mutVariableInt(individual, pool, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param pool: Sequence of arrays of length individual containing the integers
                to sample from.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if len(pool) < size:
        raise IndexError("Pool must be at least the size of individual: %d < %d" % (len(pool), size))

    for i, j in zip(range(size), pool):
        if random.random() < indpb:
            individual[i] = random.sample(j, k=1)[0]

    return individual,


# create an operator that generates a random number from 0 to 3 representing N, E, S, W:
# toolbox.register("randomDirection", random.choices, range(4), k=len(roundtrip))
toolbox.register("randomDirection", restr_direction, dirs=roundtrip.possible_dirs)

# create the individual creation operator to fill up an Individual instance with shuffled indices:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomDirection)

# create the population creation operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation - compute the total distance of the list of cities represented by indices:
def rtrp_distance(individual):
    return roundtrip.get_longest_sequence(individual)[0],  # return a tuple


toolbox.register("evaluate", rtrp_distance)

# Genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
# toolbox.register("select", tools.selRoulette)  # extremely slow
toolbox.register("mate", tools.cxOnePoint)  # 80
# toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=1.0/len(roundtrip))  # 72
toolbox.register("mutate", mutVariableInt, pool=roundtrip.possible_dirs, indpb=1.5 / len(roundtrip))


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism_plus.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True,
                                                           stop=len(roundtrip), stuck=(50, 'chernobyl'))

    # population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
    #                                                        ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,
    #                                                        verbose=True)

    # print hall of fame members info:
    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])

    # print best individual info:
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # plot best solution:
    roundtrip.plotData(best)

    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    plt.figure(1)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots:
    plt.show()


if __name__ == "__main__":

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    main()
