from typing import Callable

from deap import tools
from deap import cma

from optimizer.i_optimizer import IOptimizer
from optimizer.creator_classes import Individual
from tools import algorithms
from tools.configurations import OptimizerCmaEsCfg
from tools.helper import get_checkpoint


class OptimizerCmaEs(IOptimizer[OptimizerCmaEsCfg]):

    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: OptimizerCmaEsCfg, stats,
                 map_func=map, from_checkpoint=None):
        super(OptimizerCmaEs, self).__init__(eval_fitness, individual_size, random_seed, conf, stats, map_func,
                                             from_checkpoint)
        toolbox = self.toolbox

        if from_checkpoint:
            cp = get_checkpoint(from_checkpoint)
            toolbox.initial_generation = cp["generation"] + 1
            self.hof = cp["halloffame"]
            toolbox.recorded_individuals = cp["recorded_individuals"]
            toolbox.logbook = cp["logbook"]
            toolbox.initial_seed = cp["last_seed"]
            toolbox.strategy = cp["strategy"]
        else:
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

        toolbox.register("generate", toolbox.strategy.generate, Individual)
        toolbox.register("update", toolbox.strategy.update)
        toolbox.register("strip_strategy_from_population", self.strip_strategy_from_population,
                         mutation_learned=False)

    def train(self, number_generations) -> tools.Logbook:
        return algorithms.eaGenerateUpdate(self.toolbox, ngen=number_generations, halloffame=self.hof)
