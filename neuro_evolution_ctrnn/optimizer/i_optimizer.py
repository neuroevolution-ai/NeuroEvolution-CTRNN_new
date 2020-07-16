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

ConfigClass = TypeVar('ConfigClass', bound=IOptimizerCfg)


class IOptimizer(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed:int, conf: ConfigClass, stats, map_func=map,
                 from_checkoint=None):
        pass

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

    @staticmethod
    def create_logbook(conf: IOptimizerCfg):
        logbook = tools.Logbook()
        logbook.chapters["fitness"].header = "min", "avg", "std", "max"
        logbook.chapters["fitness"].columns_len = [8] * 4

        if conf.novelty:
            logbook.header = "gen", "nevals", "fitness", "novelty"
            logbook.columns_len = [3, 3, 0, 0]
            logbook.chapters["novelty"].header = "min", "avg", "std", "max"
            logbook.chapters["novelty"].columns_len = [8] * 4
        else:
            logbook.columns_len = [3, 3, 0]
            logbook.header = "gen", "nevals", "fitness"

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
