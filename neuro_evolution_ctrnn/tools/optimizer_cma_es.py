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


class OptimizerCmaEs(object):

    # noinspection PyUnresolvedReferences
    def __init__(self, eval_fitness: Callable, individual_size: int, conf: OptimizerCmaEsCfg, stats, map_func=map,
                 hof: tools.HallOfFame = tools.HallOfFame(5), from_checkoint=None):

        self.toolbox = toolbox = base.Toolbox()
        self.conf: OptimizerCmaEsCfg = conf
        self.toolbox.stats = stats
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)
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
            toolbox.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
            toolbox.strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=conf.sigma,
                                            lambda_=conf.population_size)
        toolbox.register("map", map_func)
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("generate", toolbox.strategy.generate, creator.Individual)
        toolbox.register("update", toolbox.strategy.update)

        cp_base_path = "checkpoints"
        Path(cp_base_path).mkdir(parents=True, exist_ok=True)
        logging.info("writing checkpoints to: " + str(os.path.abspath(cp_base_path)))
        toolbox.register("checkpoint", write_checkpoint, cp_base_path, conf.checkpoint_frequency)

    def train(self, number_generations) -> tools.Logbook:
        return algorithms.eaGenerateUpdate(self.toolbox, ngen=number_generations, halloffame=self.hof)
