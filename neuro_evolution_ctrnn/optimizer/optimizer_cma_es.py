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
from optimizer.i_optimizer import IOptimizer
from tools.helper import get_checkpoint


class OptimizerCmaEs(IOptimizer[OptimizerCmaEsCfg]):
    @staticmethod
    def create_classes():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: OptimizerCmaEsCfg, stats,
                 map_func=map,
                 hof: tools.HallOfFame = tools.HallOfFame(5), from_checkoint=None):
        super(OptimizerCmaEs, self).__init__(eval_fitness, individual_size, random_seed, conf, stats, map_func,
                                             from_checkoint)
        toolbox = self.toolbox

        if from_checkoint:
            cp = get_checkpoint(from_checkoint)
            toolbox.initial_generation = cp["generation"] + 1
            self.hof = cp["halloffame"]
            toolbox.recorded_individuals = cp["recorded_individuals"]
            toolbox.logbook = cp["logbook"]
            toolbox.initial_seed = cp["last_seed"]
            toolbox.strategy = cp["strategy"]
        else:
            self.hof = hof
            toolbox.recorded_individuals = []
            toolbox.initial_generation = 0
            toolbox.initial_seed = random_seed
            toolbox.logbook = tools.Logbook()
            toolbox.logbook = self.create_logbook(conf)
            if conf.mu:
                mu = conf.mu
                assert conf.population_size >= mu, "The population size must be higher or equal to the chosen mu."
            else:
                mu = conf.population_size // 2 if conf.population_size // 2 > 0 else 1
            toolbox.strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=conf.sigma,
                                            lambda_=conf.population_size, mu=mu)

        toolbox.register("generate", toolbox.strategy.generate, creator.Individual)
        toolbox.register("update", toolbox.strategy.update)
        toolbox.register("strip_strategy_from_population", self.strip_strategy_from_population,
                         mutation_learned=False)

    def train(self, number_generations) -> tools.Logbook:
        return algorithms.eaGenerateUpdate(self.toolbox, ngen=number_generations, halloffame=self.hof)
