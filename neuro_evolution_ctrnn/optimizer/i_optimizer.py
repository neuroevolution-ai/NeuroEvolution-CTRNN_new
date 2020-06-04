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
    def __init__(self, eval_fitness: Callable, individual_size: int, conf: ConfigClass, stats, map_func=map,
                 hof: tools.HallOfFame = tools.HallOfFame(5), from_checkoint=None):
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
