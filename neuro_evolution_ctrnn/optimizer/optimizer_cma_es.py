from typing import Callable, Collection
from deap import tools, cma
from optimizer.i_optimizer import IOptimizer
from optimizer.creator_classes import Individual
from tools.configurations import OptimizerCmaEsCfg
from tools.helper import get_checkpoint
from copy import deepcopy


class OptimizerCmaEs(IOptimizer[OptimizerCmaEsCfg]):

    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: OptimizerCmaEsCfg, stats,
                 map_func=map, from_checkpoint=None, reset_hof=False):
        super(OptimizerCmaEs, self).__init__(eval_fitness, random_seed, conf, stats, map_func,
                                             from_checkpoint, reset_hof)

        if from_checkpoint:
            # TODO get_checkpoint(from_checkpoint) is already called in super()
            self.strategy = get_checkpoint(from_checkpoint)["strategy"]
        else:
            if conf.mu:
                mu = conf.mu
                assert conf.population_size >= mu, "The population size must be higher or equal to the chosen mu."
            else:
                mu = conf.population_size // 2 if conf.population_size // 2 > 0 else 1
            self.strategy = cma.Strategy(centroid=[0.0] * individual_size, sigma=conf.sigma,
                                            lambda_=conf.population_size, mu=mu)

        self.update = self.strategy.update
        self.clone = deepcopy


    def train(self, number_generations) -> tools.Logbook:
        verbose = __debug__
        gen = None
        current_seed = None
        for gen in range(self.initial_generation, number_generations):
            population: Collection = self.strategy.generate(Individual)
            for ind in population:
                # just a little extra for when we later want to analyze the hof manually
                ind.generation = gen
            self.record_individuals(population)
            nevals, total_steps, current_seed, population = self.evaluate_candidates(population)
            if self.hof is not None:
                self.hof.update(population)
            self.update(population)
            record: dict = self.stats.compile(population) if self.stats is not None else {}

            validation_fitness, validation_steps = self.validation_runs(population)
            self.logbook.record(gen=gen, nevals=nevals, steps=total_steps + validation_steps,
                                   val_fit=validation_fitness, **record)
            if verbose:
                print(self.logbook.stream)

            self.write_checkpoint(data=dict(generation=gen, halloffame=self.hof,
                                             logbook=self.logbook, last_seed=current_seed, strategy=self.strategy,
                                             recorded_individuals=self.recorded_individuals))

        self.final_checkpoint_data = dict(generation=gen, halloffame=self.hof,
                                             logbook=self.logbook, last_seed=current_seed, strategy=self.strategy,
                                             recorded_individuals=self.recorded_individuals)
        return self.logbook
