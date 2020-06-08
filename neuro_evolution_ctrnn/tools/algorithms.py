import random

import numpy as np
from tools.helper import set_random_seeds
from typing import Iterable, Collection
from deap.algorithms import varOr
from deap import tools


def eaMuPlusLambda(toolbox, ngen, halloffame=None, verbose=__debug__,
                   include_parents_in_next_generation=True):
    population = toolbox.population

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    seed_after_map: int = random.randint(1, 10000)
    seeds_for_evaluation: np.ndarray = np.random.randint(1, 10000, size=len(invalid_ind))
    results = toolbox.map(toolbox.evaluate, invalid_ind, seeds_for_evaluation)
    for ind, res in zip(invalid_ind, results):
        ind.fitness.values = [res[0]]
        ind.behavior = res[1]
    set_random_seeds(seed_after_map, env=None)
    toolbox.recorded_individuals = []

    for gen in range(toolbox.initial_generation, ngen + 1):
        offspring = varOr(population, toolbox, toolbox.lambda_, toolbox.cxpb, toolbox.mutpb)

        if include_parents_in_next_generation:
            candidates = population + offspring
        else:
            candidates = offspring

        seed_after_map: int = random.randint(1, 10000)

        seed_for_generation = random.randint(1, 10000)
        seeds_for_evaluation = np.ones(len(candidates), dtype=np.int64) * seed_for_generation
        seeds_for_recorded = np.ones(len(toolbox.recorded_individuals), dtype=np.int64) * seed_for_generation
        results = toolbox.map(toolbox.evaluate, candidates, seeds_for_evaluation)

        results_recorded = toolbox.map(toolbox.evaluate, toolbox.recorded_individuals, seeds_for_recorded)

        for ind, res in zip(candidates, results):
            ind.fitness.values = [res[0]]
            min_distance = 10e10
            for rec_res in results_recorded:
                dist = get_behavioral_dist(res[1], rec_res[1])
                if dist < min_distance:
                    min_distance = dist
            ind.novelty = [min_distance]

        set_random_seeds(seed_after_map, env=None)
        novel_candidates = toolbox.select(candidates, toolbox.novel_base, fit_attr="novelty")
        toolbox.recorded_individuals.append(tools.selBest(novel_candidates, 1, fit_attr="fitness")[0])

        # drop recorded_individuals, when there are too many
        overfill = len(toolbox.recorded_individuals) - toolbox.max_recorded_behaviors
        if overfill > 0:
            toolbox.recorded_individuals = toolbox.recorded_individuals[overfill:]

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(candidates, toolbox.mu / 2) + toolbox.select(novel_candidates, toolbox.mu / 2)

        record = toolbox.stats.compile(population) if toolbox.stats is not None else {}
        toolbox.logbook.record(gen=gen, nevals=len(candidates), **record)
        if verbose:
            print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame, population=population,
                                         logbook=toolbox.logbook, last_seed=seed_after_map, strategy=None,
                                         recorded_behaviors=toolbox.recorded_individuals))

    return toolbox.logbook


def get_behavioral_dist(a, b):
    b = np.array(b).flatten()
    a = np.array(a).flatten()
    x = min(len(a), len(b))
    b = b[:x]
    a = a[:x]
    return np.linalg.norm(a - b)


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
