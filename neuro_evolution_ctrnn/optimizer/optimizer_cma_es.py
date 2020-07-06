import logging
import os
from pathlib import Path
from typing import Callable
from deap import base
from deap import creator
from deap import tools
from deap import cma
from tools import algorithms
from tools.configurations import OptimizerCmaEsCfg
from tools.helper import write_checkpoint, get_checkpoint
from optimizer.i_optimizer import IOptimizer


class OptimizerCmaEs(IOptimizer[OptimizerCmaEsCfg]):
    @staticmethod
    def create_classes():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

    def __init__(self, eval_fitness: Callable, individual_size: int, conf: OptimizerCmaEsCfg, stats, map_func=map,
                 hof: tools.HallOfFame = tools.HallOfFame(5), from_checkoint=None):
        super(OptimizerCmaEs, self).__init__(eval_fitness, individual_size, conf, stats, map_func,
                                              from_checkoint)
        self.toolbox = toolbox = base.Toolbox()
        self.conf: OptimizerCmaEsCfg = conf
        self.toolbox.stats = stats
        self.create_classes()
        if from_checkoint:
            cp = get_checkpoint(from_checkoint)
            toolbox.initial_generation = cp["generation"] + 1
            self.hof = cp["halloffame"]
            toolbox.logbook = cp["logbook"]
            toolbox.initial_seed = cp["last_seed"]
            toolbox.strategy = cp["strategy"]
        else:
            self.hof = hof
            toolbox.initial_generation = 0
            toolbox.initial_seed = None
            toolbox.logbook = tools.Logbook()
            toolbox.logbook = self.create_logbook(conf)
            toolbox.strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=conf.sigma,
                                            lambda_=conf.population_size)
        toolbox.register("map", map_func)
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("generate", toolbox.strategy.generate, creator.Individual)
        toolbox.register("update", toolbox.strategy.update)

        self.register_checkpoints(toolbox, conf.checkpoint_frequency)

    def train(self, number_generations) -> tools.Logbook:
        return algorithms.eaGenerateUpdate(self.toolbox, ngen=number_generations, halloffame=self.hof)
