import numpy as np

from deap import base
from deap import creator
from deap import tools

# constants:
DIMENSIONS = 2
POPULATION_SIZE = 20
MAX_GENERATIONS = 500
MIN_START_POSITION, MAX_START_POSITION = -5, 5
MIN_SPEED, MAX_SPEED = -3, 3
MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0

# set the random seed:
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# define the particle class based on ndarray:
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None)

# create and initialize a new particle:
def createParticle():
    particle = creator.Particle(np.random.uniform(MIN_START_POSITION,
                                                  MAX_START_POSITION,
                                                  DIMENSIONS))
    particle.speed = np.random.uniform(MIN_SPEED, MAX_SPEED, DIMENSIONS)
    return particle

# create the 'particleCreator' operator to fill up a particle instance:
toolbox.register("particleCreator", createParticle)


# create the 'population' operator to generate a list of particles:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)


def updateParticle(particle, best):

    # create random factors:
    localUpdateFactor = np.random.uniform(0, MAX_LOCAL_UPDATE_FACTOR, particle.size)
    globalUpdateFactor = np.random.uniform(0, MAX_GLOBAL_UPDATE_FACTOR, particle.size)

    # calculate local and global speed updates:
    localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    globalSpeedUpdate = globalUpdateFactor * (best - particle)

    # scalculate updated speed:
    particle.speed = particle.speed + (localSpeedUpdate + globalSpeedUpdate)

    # enforce limits on the updated speed:
    particle.speed = np.clip(particle.speed, MIN_SPEED, MAX_SPEED)

    # replace particle position with old-position + speed:
    particle[:] = particle + particle.speed


toolbox.register("update", updateParticle)


# Himmelblau function:
def himmelblau(particle):
    x = particle[0]
    y = particle[1]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f,  # return a tuple


toolbox.register("evaluate", himmelblau)


def main():
    # create the population of particle population:
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for generation in range(MAX_GENERATIONS):

        # evaluate all particles in polulation:
        for particle in population:

            # find the fitness of the particle:
            particle.fitness.values = toolbox.evaluate(particle)

            # particle best needs to be updated:
            if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values

            # global best needs to be updated:
            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        # update each particle's speed and position:
        for particle in population:
            toolbox.update(particle, best)

        # record the statistics for the current generation and print it:
        logbook.record(gen=generation, evals=len(population), **stats.compile(population))
        print(logbook.stream)

    # print info for best solution found:
    print("-- Best Particle = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])


if __name__ == "__main__":
    main()