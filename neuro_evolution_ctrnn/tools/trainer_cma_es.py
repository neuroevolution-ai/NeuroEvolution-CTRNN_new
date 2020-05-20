from deap import base
from deap import creator
from deap import tools
from deap import cma
from tools import algorithms

from collections import namedtuple
from typing import Callable
from tools.configurations import TrainerCmaEsCfg


class TrainerCmaEs(object):

    # noinspection PyUnresolvedReferences
    def __init__(self, eval_fitness: Callable, individual_size: int, conf: TrainerCmaEsCfg, map_func=map,
                 hof: tools.HallOfFame = tools.HallOfFame(5)):
        self.toolbox = toolbox = base.Toolbox()
        self.conf = conf
        self.hof = hof
        self.toolbox.cb_before_each_generation = None

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
        toolbox.strategy = strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=conf.sigma,
                                                   lambda_=conf.population_size)
        toolbox.register("map", map_func)
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

    def train(self, stats, number_generations) -> (object, tools.Logbook):
        return algorithms.eaGenerateUpdate(self.toolbox, ngen=number_generations,
                                           stats=stats, halloffame=self.hof)
