import abc
import copy
import logging
import os
import pickle

from deap import tools
from pathlib import Path
from typing import TypeVar, Generic, Callable, Collection
from tools.helper import get_checkpoint
from tools.configurations import IOptimizerCfg
import random
import numpy as np
from tools.helper import set_random_seeds
from functools import partial

ConfigClass = TypeVar("ConfigClass", bound=IOptimizerCfg)


class IOptimizer(abc.ABC, Generic[ConfigClass]):
    FIX_SEED_PER_GENERATION = "per_generation"
    FIX_SEED_GLOBAL = "global"
    FIX_SEED_FALSE = "false"

    @abc.abstractmethod
    def __init__(self, eval_fitness: Callable, random_seed: int, conf: ConfigClass, stats,
                 map_func=map, from_checkpoint=None, reset_hof=False):
        self.conf: ConfigClass = conf
        self.stats = stats
        self.conf = conf
        self.map = map_func
        self.evaluate = eval_fitness
        self.shape_fitness = self.shape_fitness_weighted_ranks

        # register checkpoints
        cp_base_path = "checkpoints"
        Path(cp_base_path).mkdir(parents=True, exist_ok=True)
        logging.info("writing checkpoints to: " + str(os.path.abspath(cp_base_path)))
        self.cp_base_path = cp_base_path
        self.get_distance = lambda *args: 0

        self.reset_hof = reset_hof
        if conf.novelty and not conf.fix_seed_for_generation:
            logging.warning("When using novelty you should also set fix_seed_for_generation to true. ")

        self.initial_seed = None
        if from_checkpoint:
            cp = get_checkpoint(from_checkpoint)
            self.initial_generation = cp["generation"] + 1
            self.initial_seed = cp["last_seed"]
            self.logbook = cp["logbook"]
            self.recorded_individuals = cp["recorded_individuals"]
            if not reset_hof:
                self.hof = cp["halloffame"]
        else:
            self.hof = tools.HallOfFame(self.conf.hof_size)
            self.recorded_individuals = []
            self.initial_generation = 0
            self.initial_seed = random_seed

            # create logbook
            self.logbook = tools.Logbook()
            self.logbook.chapters["fitness"].header = "min", "avg", "std", "max"
            self.logbook.chapters["fitness"].columns_len = [8] * 4
            header = "gen", "nevals", "steps", "val_fit", "fitness"
            columns_len = [4, 4, 4, 8, 0]
            if conf.novelty:
                header = header + ("novelty",)
                columns_len = columns_len + [0]
                self.logbook.chapters["novelty"].header = "min", "avg", "std", "max"
                self.logbook.chapters["novelty"].columns_len = [8] * 4
            self.logbook.columns_len = columns_len
            self.logbook.header = header

    @abc.abstractmethod
    def train(self, number_generations) -> tools.Logbook:
        pass

    def shape_fitness_multi_objective(self, population):
        for ind in population:
            ind.fitness_orig = copy.deepcopy(ind.fitness)
            shaped_fitness = [ind.fitness.values[0]]
            novelty = ind.novelty if self.conf.novelty else 0
            efficiency = -ind.steps if self.conf.efficiency_weight else 0
            shaped_fitness.append(novelty)
            shaped_fitness.append(efficiency)
            ind.fitness.values = tuple(shaped_fitness)

    def shape_fitness_weighted_ranks(self, population):
        MINIMUM = -10e10
        if self.conf.novelty:
            novel_counter = 0
            novel_last = MINIMUM
            for ind in sorted(population, key=lambda x: x.novelty):
                if novel_last != ind.novelty:
                    novel_counter += 1
                novel_last = ind.novelty
                ind.novelty_rank = novel_counter

        if self.conf.efficiency_weight:
            efficiency_counter = 0
            efficiency_last = MINIMUM
            for ind in sorted(population, key=lambda x: -x.steps):
                if efficiency_last != ind.steps:
                    efficiency_counter += 1
                efficiency_last = ind.steps
                ind.efficiency_rank = efficiency_counter

        fitness_counter = 0
        fitness_last = MINIMUM
        for ind in sorted(population, key=lambda x: x.fitness_orig):
            if fitness_last != ind.fitness_orig:
                fitness_counter += 1
            fitness_last = ind.fitness_orig
            ind.fitness_rank = fitness_counter

        for ind in population:
            # assert not hasattr(ind, 'fitness_orig'), 'individual already has shaped fitness'

            shaped_fitness = ind.fitness_rank
            if self.conf.novelty:
                shaped_fitness += self.conf.novelty.novelty_weight * ind.novelty_rank
            if self.conf.efficiency_weight:
                shaped_fitness += self.conf.efficiency_weight * ind.efficiency_rank
            ind.fitness.values = tuple([shaped_fitness])

    def evaluate_candidates(self, candidates):
        if self.initial_seed:
            seed_for_generation = None
            if self.conf.fix_seed == IOptimizer.FIX_SEED_FALSE:
                seeds_for_evaluation: np.ndarray = np.random.randint(1, 10000, size=len(candidates))
                seeds_for_recorded: np.ndarray = np.random.randint(1, 10000,
                                                                   size=len(self.recorded_individuals))
            elif self.conf.fix_seed == IOptimizer.FIX_SEED_PER_GENERATION:
                seed_for_generation = np.random.randint(1, 10000)
                seeds_for_evaluation: np.ndarray = seed_for_generation * np.ones(len(candidates), dtype=np.int64)
                seeds_for_recorded = seed_for_generation * np.ones(len(self.recorded_individuals),
                                                                   dtype=np.int64)
            elif self.conf.fix_seed == IOptimizer.FIX_SEED_GLOBAL:
                seed_for_generation = self.initial_seed
                seeds_for_evaluation: np.ndarray = seed_for_generation * np.ones(len(candidates), dtype=np.int64)
                seeds_for_recorded = seed_for_generation * np.ones(len(self.recorded_individuals),
                                                                   dtype=np.int64)
            else:
                raise NotImplementedError()
        else:
            seed_for_generation = 0
            seeds_for_evaluation = np.zeros(len(candidates), dtype=np.int64)
            seeds_for_recorded = np.zeros(len(self.recorded_individuals), dtype=np.int64)

        brain_genomes_recorded = self.strip_strategy_from_population(self.recorded_individuals)
        nevals = len(candidates) + len(brain_genomes_recorded)

        novelty_data = (
                           [(list(self.map(self.evaluate, brain_genomes_recorded, seeds_for_recorded)),
                             self.get_distance,
                             self.conf.novelty.novelty_nearest_k)] if self.conf.novelty else [None]
                       ) * len(candidates)

        candidates, total_steps = zip(
            *self.map(
                self.evaluate,
                candidates,
                seeds_for_evaluation,
                novelty_data,
                [self.get_number_genes_for_strategy(len(candidates[0]))] * len(candidates)
            )
        )
        candidates = list(candidates)
        total_steps = sum(total_steps)
        # setting seeds must happen after reading all fitnesses from results, because of the async nature of map, it
        # is possiblethat some evaluations are still running when the first results get processes
        set_random_seeds(seed_for_generation, env=None)

        self.shape_fitness(candidates)

        if self.conf.novelty:
            # drop recorded_individuals, when there are too many
            if self.conf.novelty.max_recorded_behaviors:
                overfill = len(self.recorded_individuals) - self.conf.novelty.max_recorded_behaviors
                if overfill > 0:
                    self.recorded_individuals = self.recorded_individuals[overfill:]

        return nevals, total_steps, seed_for_generation, candidates

    def record_individuals(self, population):
        if self.conf.novelty:
            self.recorded_individuals += random.choices(
                population, k=self.conf.novelty.recorded_behaviors_per_generation)

    def validation_runs(self, population):
        intial_seed = self.conf.validation_initial_seed
        number_validation_runs = self.conf.validation_number_runs

        if number_validation_runs < 1:
            return 0, 0
        seeds_validation_runs: np.ndarray = np.array(range(intial_seed, intial_seed + number_validation_runs))
        best_individuals = tools.selBest(population, 1, fit_attr="fitness")  # only validate the single best individual

        brain_genomes = self.strip_strategy_from_population(best_individuals)
        results = []
        for genome in brain_genomes:
            # partial is needed because keyword argument
            # see https://stackoverflow.com/a/13499853/5132456
            evaluate = partial(self.evaluate, rounds=1)
            results += self.map(evaluate, [genome] * number_validation_runs, seeds_validation_runs)
        total_steps = 0
        sum_fitness = 0.0
        for res in results:
            fitness, behavior_compressed, steps = res
            total_steps += steps
            sum_fitness += fitness
        return sum_fitness / number_validation_runs, total_steps

    @classmethod
    def strip_strategy_from_population(cls, population, strategy_parameter_per_gene=False):
        """Sometimes strategy parameters are learned along side brain parameters. In these caeses
        the strategy parameters need to be stripped  from the population before sending the brain genomes to
        the evaluation. """
        return population

    def get_number_genes_for_strategy(self, individual_size):
        return 0

    @classmethod
    def strip_strategy_from_individual(cls, individual, strategy_parameter_per_gene=False):
        return individual

    def write_checkpoint(self, data):
        if not self.conf.checkpoint_frequency:
            return
        if data["generation"] % self.conf.checkpoint_frequency != 0:
            return

        filename = os.path.join(self.cp_base_path, "checkpoint_" + str(data["generation"]) + ".pkl")
        logging.info("Writing checkpoint " + filename)
        with open(filename, "wb") as cp_file:
            pickle.dump(data, cp_file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)