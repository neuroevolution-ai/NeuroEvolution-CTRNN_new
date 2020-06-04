import random

import numpy as np
from tools.helper import set_random_seeds
from typing import Iterable, Collection
from deap.algorithms import varOr


def eaMuPlusLambda(toolbox, ngen, halloffame=None, verbose=__debug__,
                   include_parents_in_next_generation=True):
    population = toolbox.population

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    seed_after_map: int = random.randint(1, 10000)
    seeds_for_evaluation: np.ndarray = np.random.randint(1, 10000, size=len(invalid_ind))
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, seeds_for_evaluation)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    set_random_seeds(seed_after_map, env=None)

    for gen in range(toolbox.initial_generation, ngen + 1):
        offspring = varOr(population, toolbox, toolbox.lambda_, toolbox.cxpb, toolbox.mutpb)

        if include_parents_in_next_generation:
            candidates = population + offspring
        else:
            candidates = offspring

        seed_after_map: int = random.randint(1, 10000)
        seeds_for_evaluation: np.ndarray = np.random.randint(1, 10000, size=len(candidates))
        fitnesses = toolbox.map(toolbox.evaluate, candidates, seeds_for_evaluation)
        for ind, fit in zip(candidates, fitnesses):
            ind.fitness.values = fit
        set_random_seeds(seed_after_map, env=None)

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(candidates, toolbox.mu)

        record = toolbox.stats.compile(population) if toolbox.stats is not None else {}
        toolbox.logbook.record(gen=gen, nevals=len(candidates), **record)
        if verbose:
            print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame,
                                         logbook=toolbox.logbook, last_seed=seed_after_map, strategy=None))

    return toolbox.logbook


def eaGenerateUpdate(toolbox, ngen: int, halloffame=None):
    if toolbox.initial_seed:
        set_random_seeds(toolbox.initial_seed, env=None)

    for gen in range(toolbox.initial_generation, ngen + 1):
        population: Collection = toolbox.generate()
        seed_after_map: int = random.randint(1, 10000)
        seeds_for_evaluation: np.ndarray = np.random.randint(1, 10000, size=len(population))
        finesses: Iterable = toolbox.map(toolbox.evaluate, population, seeds_for_evaluation)
        for ind, fit in zip(population, finesses):
            ind.fitness.values = fit
        # reseed because workers seem to affect the global state
        # also this must happen AFTER fitness-values have been processes, because futures
        set_random_seeds(seed_after_map, env=None)
        if halloffame is not None:
            halloffame.update(population)
        toolbox.update(population)
        record: dict = toolbox.stats.compile(population)
        toolbox.logbook.record(gen=gen, nevals=len(population), **record)
        print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame,
                                         logbook=toolbox.logbook, last_seed=seed_after_map, strategy=toolbox.strategy))

    return toolbox.logbook
