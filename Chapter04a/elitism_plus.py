from deap import tools
from deap import algorithms
import random


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, stop=0, stuck=(1e9, None)):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    stuck_count = 0
    last_min = False

    save_mutpb = mutpb
    radiation = 0

    # Begin the generational process
    for gen in range(1, ngen + 1):

        radiation = max(radiation - 1, 0)
        if radiation == 0:
            mutpb = save_mutpb
        else:
            stuck_count = 0
            print(f'radiation is {radiation}')

        # Select the next generation individuals
        if stuck[0] < stuck_count:
            if stuck[1] == 'comet':
                # Generate new population for non-hof (Comet-Strike)
                print('the comet strikes')
                offspring = toolbox.populationCreator(len(population) - 3)
                offspring.extend(halloffame.items[:3])
                halloffame.clear()
            if stuck[1] == 'chernobyl':
                print('radiation leak')
                mutpb = 0.5
                radiation = stuck[0]
            stuck_count = 0
        else:
            # Use defined selection algorithm
            offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Check if minimum has change vs previous iteration, else rais stuck_count
        new_min = min(logbook.select('min'))

        if last_min:
            if new_min == last_min:
                stuck_count += 1
            else:
                stuck_count = 0

        if radiation == 0:
            print(f'stuck count is {stuck_count}')

        last_min = new_min

        # early stopping, if zero is reached (optimum)
        if last_min == stop:
            break

    return population, logbook

