import random

import numpy as np
from tools.helper import set_random_seeds
from typing import Collection
from deap.algorithms import varOr
from deap import tools
from bz2 import decompress
from itertools import tee
from functools import partial


def evaluate_candidates(candidates, toolbox):
    if toolbox.initial_seed:
        seed_after_map: int = np.random.randint(1, 10000)
        if toolbox.conf.fix_seed_for_generation:
            seed_for_generation = np.random.randint(1, 10000)
            seeds_for_evaluation: np.ndarray = seed_for_generation * np.ones(
                len(candidates), dtype=np.int64
            )
            seeds_for_recorded = seed_for_generation * np.ones(
                len(toolbox.recorded_individuals), dtype=np.int64
            )
        else:
            seeds_for_evaluation: np.ndarray = np.random.randint(
                1, 10000, size=len(candidates)
            )
            seeds_for_recorded: np.ndarray = np.random.randint(
                1, 10000, size=len(candidates)
            )
    else:
        seed_after_map = 0
        seeds_for_evaluation = np.zeros(len(candidates), dtype=np.int64)
        seeds_for_recorded = np.zeros(len(toolbox.recorded_individuals), dtype=np.int64)

    brain_genomes = toolbox.strip_strategy_from_population(candidates)
    brain_genomes_recorded = toolbox.strip_strategy_from_population(
        toolbox.recorded_individuals
    )
    nevals = len(brain_genomes) + len(brain_genomes_recorded)
    results = toolbox.map(toolbox.evaluate, brain_genomes, seeds_for_evaluation)

    if toolbox.conf.novelty:
        results_recorded_orig = list(
            toolbox.map(toolbox.evaluate, brain_genomes_recorded, seeds_for_recorded)
        )
        results_copy, results = tee(results, 2)
        novelties = toolbox.map(
            calc_novelty,
            list(results_copy),
            [results_recorded_orig] * len(candidates),
            [toolbox.get_distance] * len(candidates),
            [toolbox.conf.novelty.novelty_nearest_k] * len(candidates),
        )
    else:
        novelties = [0] * len(candidates)

    total_steps = 0
    for ind, res, nov in zip(candidates, results, novelties):
        fitness, behavior_compressed, steps = res
        ind.fitness_orig = fitness
        ind.novelty = nov
        ind.steps = steps
        total_steps += steps
    # setting seeds must happen after reading all fitnesses from results, because of the async nature of map, it
    # is possiblethat some evaluations are still running when the first results get processes
    set_random_seeds(seed_after_map, env=None)

    toolbox.shape_fitness(candidates)

    if toolbox.conf.novelty:
        # drop recorded_individuals, when there are too many
        if toolbox.conf.novelty.max_recorded_behaviors:
            overfill = (
                len(toolbox.recorded_individuals)
                - toolbox.conf.novelty.max_recorded_behaviors
            )
            if overfill > 0:
                toolbox.recorded_individuals = toolbox.recorded_individuals[overfill:]

    return nevals, total_steps, seed_after_map


def record_individuals(toolbox, population):
    if toolbox.conf.novelty:
        toolbox.recorded_individuals += random.choices(
            population, k=toolbox.conf.novelty.recorded_behaviors_per_generation
        )


def validation_runs(population, toolbox):
    intial_seed = toolbox.conf.validation_initial_seed
    number_validation_runs = toolbox.conf.validation_number_runs

    if number_validation_runs < 1:
        return 0, 0
    seeds_validation_runs: np.ndarray = np.array(
        range(intial_seed, intial_seed + number_validation_runs)
    )
    best_individuals = tools.selBest(
        population, 1, fit_attr="fitness"
    )  # only validate the single best individual

    brain_genomes = toolbox.strip_strategy_from_population(best_individuals)
    results = []
    for genome in brain_genomes:
        # partial is needed because keyword argument
        # see https://stackoverflow.com/a/13499853/5132456
        evaluate = partial(toolbox.evaluate, rounds=1)
        results += toolbox.map(
            evaluate, [genome] * number_validation_runs, seeds_validation_runs
        )
    total_steps = 0
    sum_fitness = 0.0
    for res in results:
        fitness, behavior_compressed, steps = res
        total_steps += steps
        sum_fitness += fitness
    return sum_fitness / number_validation_runs, total_steps


def eaMuPlusLambda(
    toolbox, ngen, verbose=__debug__, include_parents_in_next_generation=True
):
    population = toolbox.population
    halloffame = toolbox.hof
    gen = -1
    current_seed = -1

    for gen in range(toolbox.initial_generation, ngen):
        record_individuals(toolbox, population)
        extra = []
        if halloffame.items:
            extra = list(
                map(
                    toolbox.clone,
                    random.sample(population, toolbox.conf.extra_from_hof),
                )
            )
        offspring = varOr(
            population + extra,
            toolbox,
            toolbox.conf.lambda_,
            1 - toolbox.conf.mutpb,
            toolbox.conf.mutpb,
        )

        for ind in offspring:
            # just a little extra for when we later want to analyze the hof manually
            ind.generation = gen

        if include_parents_in_next_generation:
            for ind in population:
                del ind.fitness.values
            candidates = population + offspring
        else:
            candidates = offspring

        nevals, total_steps, current_seed = evaluate_candidates(candidates, toolbox)

        if halloffame is not None:
            halloffame.update(offspring)

        record = toolbox.stats.compile(candidates) if toolbox.stats is not None else {}
        population[:] = toolbox.select(candidates, toolbox.conf.mu)
        validation_fitness, validation_steps = validation_runs(population, toolbox)
        toolbox.logbook.record(
            gen=gen,
            nevals=nevals,
            steps=total_steps + validation_steps,
            val_fit=validation_fitness,
            **record
        )
        if verbose:
            print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(
                data=dict(
                    generation=gen,
                    halloffame=halloffame,
                    population=population,
                    logbook=toolbox.logbook,
                    last_seed=current_seed,
                    strategy=None,
                    recorded_individuals=toolbox.recorded_individuals,
                )
            )
    toolbox.final_checkpoint_data = dict(
        generation=gen,
        halloffame=halloffame,
        population=population,
        logbook=toolbox.logbook,
        last_seed=current_seed,
        strategy=None,
        recorded_individuals=toolbox.recorded_individuals,
    )
    return toolbox.logbook


def calc_novelty(res, results_recorded, get_distance, k):
    """calculate the average distance to the k nearest neighbors"""
    if k < 1:
        k = len(results_recorded)
    behavior_compressed = res[1]
    behavior_data = decompress(behavior_compressed)
    behavior = np.frombuffer(behavior_data, dtype=np.float16)
    dist_list = []
    k = min(k, len(results_recorded))
    for rec_res in results_recorded:
        recorded_behavior_compressed = rec_res[1]
        recorded_behavior_data = decompress(recorded_behavior_compressed)
        recorded_behavior = np.frombuffer(recorded_behavior_data, dtype=np.float16)
        dist = get_distance(
            a=behavior,
            b=recorded_behavior,
            a_len=len(behavior_compressed),
            b_len=len(recorded_behavior_compressed),
        )
        dist_list.append(dist)

    dist_sum = 0
    for nearest_neighbor in sorted(dist_list, reverse=False)[0:k]:
        dist_sum += nearest_neighbor
    return dist_sum / k


def eaGenerateUpdate(toolbox, ngen: int, halloffame=None):
    if toolbox.initial_seed:
        # set_random_seeds(toolbox.initial_seed, env=None)
        pass
    gen = -1
    current_seed = -1

    for gen in range(toolbox.initial_generation, ngen):
        population: Collection = toolbox.generate()

        for ind in population:
            # just a little extra for when we later want to analyze the hof manually
            ind.generation = gen
        record_individuals(toolbox, population)
        nevals, total_steps, current_seed = evaluate_candidates(population, toolbox)
        if halloffame is not None:
            halloffame.update(population)
        toolbox.update(population)
        record: dict = toolbox.stats.compile(population)

        validation_fitness, validation_steps = validation_runs(population, toolbox)
        toolbox.logbook.record(
            gen=gen,
            nevals=nevals,
            steps=total_steps + validation_steps,
            val_fit=validation_fitness,
            **record
        )
        print(toolbox.logbook.stream)
        if toolbox.checkpoint:
            toolbox.checkpoint(
                data=dict(
                    generation=gen,
                    halloffame=halloffame,
                    logbook=toolbox.logbook,
                    last_seed=current_seed,
                    strategy=toolbox.strategy,
                    recorded_individuals=toolbox.recorded_individuals,
                )
            )

    toolbox.final_checkpoint_data = dict(
        generation=gen,
        halloffame=halloffame,
        logbook=toolbox.logbook,
        last_seed=current_seed,
        strategy=toolbox.strategy,
        recorded_individuals=toolbox.recorded_individuals,
    )
    return toolbox.logbook
