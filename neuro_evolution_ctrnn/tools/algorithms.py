import random

import numpy as np
from tools.helper import set_random_seeds
from typing import Iterable, Collection
from deap.algorithms import varOr
from deap import tools
from bz2 import compress, decompress
from itertools import tee


def evaluate_candidates(candidates, toolbox):
    if toolbox.initial_seed:
        seed_after_map: int = random.randint(1, 10000)
        seed_for_generation = random.randint(1, 10000)
    else:
        seed_after_map = 0
        seed_for_generation = 0

    brain_genomes = toolbox.strip_strategy_from_population(candidates)
    brain_genomes_recorded = toolbox.strip_strategy_from_population(toolbox.recorded_individuals)
    nevals = len(brain_genomes) + len(brain_genomes_recorded)

    seeds_for_evaluation = np.ones(len(brain_genomes), dtype=np.int64) * seed_for_generation
    results = toolbox.map(toolbox.evaluate, brain_genomes, seeds_for_evaluation)

    if toolbox.conf.novelty:
        seeds_for_recorded = np.ones(len(toolbox.recorded_individuals), dtype=np.int64) * seed_for_generation
        results_recorded_orig = list(toolbox.map(toolbox.evaluate, brain_genomes_recorded, seeds_for_recorded))
        results_copy, results = tee(results, 2)
        novelties = toolbox.map(calc_novelty,
                                list(results_copy),
                                [results_recorded_orig] * len(candidates),
                                [toolbox.get_distance] * len(candidates),
                                [toolbox.conf.novelty.novelty_nearest_k] * len(candidates))
    else:
        novelties = [0] * len(candidates)

    total_steps = 0
    for ind, res, nov in zip(candidates, results, novelties):
        fitness, behavior_compressed, steps = res
        ind.fitness.values = [fitness]
        ind.novelty = nov
        ind.steps = steps
        total_steps += steps
    # setting seeds must happen after reading alle fitnesses from results, because of the async nature of map, it
    # is possiblethat some evaluations are still running when the first results get processes
    set_random_seeds(seed_after_map, env=None)

    toolbox.shape_fitness(candidates)
    return nevals, total_steps, seed_after_map

def record_individuals(toolbox, population):
    if toolbox.conf.novelty:
        toolbox.recorded_individuals += random.choices(population,
                                                       k=toolbox.conf.novelty.recorded_behaviors_per_generation)

def eaMuPlusLambda(toolbox, ngen, verbose=__debug__,
                   include_parents_in_next_generation=True):
    population = toolbox.population
    halloffame = toolbox.hof

    for gen in range(toolbox.initial_generation, ngen + 1):
        record_individuals(toolbox, population)
        extra = []
        if halloffame.items:
            extra = list(map(toolbox.clone, random.sample(population, toolbox.conf.extra_from_hof)))

        offspring = varOr(population + extra, toolbox, toolbox.conf.lambda_, 1 - toolbox.conf.mutpb, toolbox.conf.mutpb)

        if include_parents_in_next_generation:
            candidates = population + offspring
        else:
            candidates = offspring

        nevals, total_steps, current_seed = evaluate_candidates(candidates, toolbox)

        if toolbox.conf.novelty:
            # drop recorded_individuals, when there are too many
            overfill = len(toolbox.recorded_individuals) - toolbox.conf.novelty.max_recorded_behaviors
            if overfill > 0:
                toolbox.recorded_individuals = toolbox.recorded_individuals[overfill:]

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(candidates, toolbox.conf.mu, fit_attr="shaped_fitness")

        record = toolbox.stats.compile(population) if toolbox.stats is not None else {}
        toolbox.logbook.record(gen=gen, nevals=nevals, steps=total_steps, **record)
        if verbose:
            print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame, population=population,
                                         logbook=toolbox.logbook, last_seed=current_seed, strategy=None,
                                         recorded_individuals=toolbox.recorded_individuals))

    return toolbox.logbook


def calc_novelty(res, results_recorded, get_distance, k):
    """calculate the average distance to the k nearest neighbors"""
    behavior_compressed = res[1]
    behavior_data = decompress(behavior_compressed)
    behavior = np.frombuffer(behavior_data, dtype=np.float16)
    dist_list = []
    for rec_res in results_recorded:
        recorded_behavior_compressed = rec_res[1]
        recorded_behavior_data = decompress(recorded_behavior_compressed)
        recorded_behavior = np.frombuffer(recorded_behavior_data, dtype=np.float16)
        dist = get_distance(a=behavior, b=recorded_behavior, a_len=len(behavior_compressed),
                            b_len=len(recorded_behavior_compressed))
        dist_list.append(dist)

    dist_sum = 0
    for nearest_neighbor in sorted(dist_list, reverse=False)[0:k]:
        dist_sum += nearest_neighbor

    return dist_sum / k


def eaGenerateUpdate(toolbox, ngen: int, halloffame=None):
    if toolbox.initial_seed:
        set_random_seeds(toolbox.initial_seed, env=None)

    for gen in range(toolbox.initial_generation, ngen + 1):
        population: Collection = toolbox.generate()
        record_individuals(toolbox, population)
        nevals, total_steps, current_seed = evaluate_candidates(population, toolbox)
        if halloffame is not None:
            halloffame.update(population)
        toolbox.update(population)
        record: dict = toolbox.stats.compile(population)
        toolbox.logbook.record(gen=gen, nevals=len(population), steps=total_steps, **record)
        print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame,
                                         logbook=toolbox.logbook, last_seed=current_seed, strategy=toolbox.strategy))

    return toolbox.logbook
