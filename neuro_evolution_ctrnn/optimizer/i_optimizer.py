import abc
from tools.configurations import IOptimizerCfg
import numpy as np
from gym.spaces import Space, Discrete, Box
from typing import TypeVar, Generic, Callable
from deap import tools
from pathlib import Path
import logging
import os
from tools.helper import write_checkpoint
from tools.helper import normalized_compression_distance, euklidian_distance, equal_elements_distance
from deap import base
import copy


ConfigClass = TypeVar('ConfigClass', bound=IOptimizerCfg)


class IOptimizer(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: ConfigClass, stats,
                 map_func=map,
                 from_checkoint=None):
        self.create_classes()
        self.conf: ConfigClass = conf
        self.toolbox = toolbox = base.Toolbox()
        self.toolbox.stats = stats
        toolbox.conf = conf
        toolbox.register("map", map_func)
        toolbox.register("evaluate", eval_fitness)
        # toolbox.register("shape_fitness", lambda *args: None)
        toolbox.register("shape_fitness", self.shape_fitness_weighted_ranks)
        self.register_checkpoints(toolbox, conf.checkpoint_frequency)
        self.register_novelty_distance(toolbox)

        if conf.novelty and not conf.fix_seed_for_generation:
            logging.warning("When using novelty you should also set fix_seed_for_generation to true. ")

    @staticmethod
    @abc.abstractmethod
    def create_classes():
        pass

    @abc.abstractmethod
    def train(self, number_generations) -> tools.Logbook:
        pass

    @staticmethod
    def register_checkpoints(toolbox, checkpoint_frequency):
        cp_base_path = "checkpoints"
        Path(cp_base_path).mkdir(parents=True, exist_ok=True)
        logging.info("writing checkpoints to: " + str(os.path.abspath(cp_base_path)))
        toolbox.register("checkpoint", write_checkpoint, cp_base_path, checkpoint_frequency)

    def register_novelty_distance(self, toolbox):
        conf = self.conf
        if conf.novelty:
            if conf.novelty.distance == "euclid":
                toolbox.register("get_distance", euklidian_distance)
            elif conf.novelty.distance == "NCD":
                toolbox.register("get_distance", normalized_compression_distance)
            elif conf.novelty.distance == "equal":
                toolbox.register("get_distance", equal_elements_distance)
            else:
                raise RuntimeError("unknown configuration value for distance: " + str(conf.novelty.distance))
        else:
            toolbox.register("get_distance", lambda *args: 0)

    @staticmethod
    def create_logbook(conf: IOptimizerCfg):
        logbook = tools.Logbook()
        logbook.chapters["fitness"].header = "min", "avg", "std", "max"
        logbook.chapters["fitness"].columns_len = [8] * 4

        if conf.novelty:
            logbook.header = "gen", "nevals", "steps", "fitness", "novelty"
            logbook.columns_len = [4, 4, 8, 0, 0]
            logbook.chapters["novelty"].header = "min", "avg", "std", "max"
            logbook.chapters["novelty"].columns_len = [8] * 4
        else:
            logbook.columns_len = [4, 4, 8, 0]
            logbook.header = "gen", "nevals", "steps", "fitness"

        return logbook

    @staticmethod
    def strip_strategy_from_population(population, mutation_learned):
        """Sometimes strategy parameters are learned along side brain parameters. In these caeses
        the strategy parameters need to be stripped  from the population before sending the brain genomes to
        the evaluation. """
        if len(population) == 0:
            return population
        if mutation_learned:
            return list(np.array(population)[:, :-2])
        return population

    def shape_fitness_multi_objective(self, population):
        for ind in population:
            ind.fitness_orig = copy.deepcopy( ind.fitness)
            shaped_fitness = [ind.fitness.values[0]]
            novelty = ind.novelty if self.conf.novelty else 0
            efficiency = -ind.steps if self.conf.efficiency_weight else 0
            shaped_fitness.append(novelty)
            shaped_fitness.append(efficiency)
            ind.fitness.values = tuple(shaped_fitness)


    def shape_fitness_weighted_ranks(self, population):

        if self.conf.novelty:
            novel_counter = 0
            for ind in sorted(population, key=lambda x: x.novelty):
                ind.novelty_rank = novel_counter
                novel_counter += 1

        if self.conf.efficiency_weight:
            efficiency_counter = 0
            for ind in sorted(population, key=lambda x: -x.steps):
                ind.efficiency_rank = efficiency_counter
                efficiency_counter += 1

        fitness_counter = 0
        for ind in sorted(population, key=lambda x: x.fitness.values[0]):
            ind.fitness_rank = fitness_counter
            fitness_counter += 1

        for ind in population:
            ind.fitness_orig = copy.deepcopy( ind.fitness)
            shaped_fitness = ind.fitness_rank
            if self.conf.novelty:
                shaped_fitness += self.conf.novelty.novelty_weight * ind.novelty_rank
            if self.conf.efficiency_weight:
                shaped_fitness += self.conf.efficiency_weight * ind.efficiency_rank
            ind.fitness.values = tuple([shaped_fitness])
