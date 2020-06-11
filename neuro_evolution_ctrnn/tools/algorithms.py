import random

import numpy as np
from tools.helper import set_random_seeds
from typing import Iterable, Collection
from deap.algorithms import varOr
from deap import tools
from bz2 import compress, decompress
from itertools import tee


def eaMuPlusLambda(toolbox, ngen, verbose=__debug__,
                   include_parents_in_next_generation=True):
    population = toolbox.population
    halloffame = toolbox.hof

    for gen in range(toolbox.initial_generation, ngen + 1):
        extra = []
        if halloffame.items:
            extra = random.choices(halloffame.items, k=toolbox.conf.extra_from_hof)
        offspring = varOr(population + extra, toolbox, toolbox.conf.lambda_, 1 - toolbox.conf.mutpb, toolbox.conf.mutpb)

        if include_parents_in_next_generation:
            candidates = population + offspring
        else:
            candidates = offspring

        seed_after_map: int = random.randint(1, 10000)
        seed_for_generation = random.randint(1, 10000)
        seeds_for_evaluation = np.ones(len(candidates), dtype=np.int64) * seed_for_generation
        seeds_for_recorded = np.ones(len(toolbox.recorded_individuals), dtype=np.int64) * seed_for_generation
        nevals = len(candidates) + len(toolbox.recorded_individuals)

        brain_genomes = toolbox.strip_strategy_from_population(candidates)
        brain_genomes_recorded = toolbox.strip_strategy_from_population(toolbox.recorded_individuals)
        results = toolbox.map(toolbox.evaluate, brain_genomes, seeds_for_evaluation)
        results_recorded_orig = list(toolbox.map(toolbox.evaluate, brain_genomes_recorded, seeds_for_recorded))

        if results_recorded_orig:
            novelties = toolbox.map(calc_novelty,
                            results,
                            [results_recorded_orig] * len(candidates),
                            [toolbox.get_distance] * len(candidates))
        else:
            novelties = [0] * len(candidates)

        for ind, res, nov in zip(candidates, results, novelties):
            fitness, behavior_compressed = res
            ind.fitness.values = [fitness]
            ind.novelty = nov

        set_random_seeds(seed_after_map, env=None)
        novel_candidates = toolbox.select(candidates, toolbox.conf.mu_mixed_base, fit_attr="novelty")
        toolbox.recorded_individuals.append(random.choice(novel_candidates))

        # drop recorded_individuals, when there are too many
        overfill = len(toolbox.recorded_individuals) - toolbox.conf.max_recorded_behaviors
        if overfill > 0:
            toolbox.recorded_individuals = toolbox.recorded_individuals[overfill:]

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(candidates, toolbox.conf.mu) + \
                        toolbox.select(novel_candidates, toolbox.conf.mu_mixed) + \
                        toolbox.select(candidates, toolbox.conf.mu_novel, fit_attr="novelty")

        record = toolbox.stats.compile(population) if toolbox.stats is not None else {}
        toolbox.logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame, population=population,
                                         logbook=toolbox.logbook, last_seed=seed_after_map, strategy=None,
                                         recorded_individuals=toolbox.recorded_individuals))

    return toolbox.logbook


def calc_novelty(res, results_recorded, get_distance):
    behavior_compressed = res[1]
    behavior = list(decompress(behavior_compressed))
    min_distance = 10e20
    for rec_res in results_recorded:
        recorded_behavior_compressed = rec_res[1]
        recorded_behavior = list(decompress(recorded_behavior_compressed))
        dist = get_distance(behavior, recorded_behavior)
        if dist < min_distance:
            min_distance = dist
    return min_distance


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
            ind.novelty = 0
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
