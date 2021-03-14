import random
from copy import deepcopy
from typing import Callable
from deap import tools
import numpy as np
from deap.algorithms import varOr
from optimizer.creator_classes import Individual
from optimizer.i_optimizer import IOptimizer
from tools.configurations import OptimizerMuLambdaCfg
from tools.helper import get_checkpoint


def sel_elitist_tournament(individuals, mu, k_elitist, k_tournament, tournsize, fit_attr="fitness"):
    return tools.selBest(individuals, int(k_elitist * mu), fit_attr="fitness") + \
           tools.selTournament(individuals, int(k_tournament * mu), tournsize=tournsize, fit_attr="fitness")


class OptimizerMuPlusLambda(IOptimizer[OptimizerMuLambdaCfg]):

    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: OptimizerMuLambdaCfg,
                 stats, map_func=map, from_checkpoint=None, reset_hof=False):
        super(OptimizerMuPlusLambda, self).__init__(eval_fitness, random_seed, conf, stats, map_func,
                                                    from_checkpoint, reset_hof)

        if self.conf.strategy_parameter_per_gene:
            individual_size *= 2
        else:
            # add two genes for strategy parameters used in mutate
            individual_size += 2

        def individual():
            def indices():
                return np.random.uniform(
                    -self.conf.initial_gene_range,
                    self.conf.initial_gene_range,
                    individual_size)

            return tools.initIterate(Individual, indices)

        self.population = tools.initRepeat(list, individual, self.conf.mu)

        mate_list = [
            tools.cxOnePoint,
            tools.cxTwoPoint,
        ]

        def mate(ind1, ind2):
            return random.choice(mate_list)(ind1, ind2)

        def fct_mutation_learned_per_gene(ind1):
            half = len(ind1) // 2
            ind1[half:] = np.clip(ind1[half:], -6, 0)
            strategy = []
            for s in ind1[half:]:
                strategy.append(2 ** s)
            # mutate genome with parameters from strategy. and mutate strategy with constant
            # sigma = np.concatenate((strategy, ([0.2] * half)), axis=None).tolist()
            sigma = np.concatenate((strategy, strategy), axis=None).tolist()
            return tools.mutGaussian(individual=ind1, mu=0, sigma=sigma, indpb=1.0)

        def fct_mutation_learned(ind1):
            # need to clip in up-direction because too large numbers create overflows
            # need to clip below to avoid stagnation, which happens when a top individuals
            # mutates bad strategy parameters
            ind1[-1] = np.clip(ind1[-1], -5, 3)
            ind1[-2] = np.clip(ind1[-2], -5, 3)
            sigma = 2 ** ind1[-1]
            indpb = 2 ** (ind1[-2] - 3)
            return tools.mutGaussian(individual=ind1, mu=0, sigma=sigma, indpb=indpb)

        self.mate = mate
        if self.conf.strategy_parameter_per_gene:
            self.mutate = fct_mutation_learned_per_gene

        else:
            self.mutate = fct_mutation_learned

        # TODO no Error?
        # self.strip_strategy_from_population = self.strip_strategy_from_population(
        #                  mutation_learned=True, strategy_parameter_per_gene=self.conf.strategy_parameter_per_gene)

        def select(inds, k):
            if self.conf.tournsize:
                return tools.selTournament(inds, k, tournsize=self.conf.tournsize)
            else:
                return tools.selBest(inds, k)
        self.select = select

        if from_checkpoint:
            # TODO get_checkpoint(from_checkpoint) is already called in super()
            self.population = get_checkpoint(from_checkpoint)["population"]
        self.clone = deepcopy

    def train(self, number_generations) -> tools.Logbook:
        verbose = __debug__
        gen = None
        current_seed = None

        for gen in range(self.initial_generation, number_generations):
            self.record_individuals(self.population)
            extra = []
            if self.hof.items:
                extra = list(map(self.clone, random.sample(self.population, self.conf.extra_from_hof)))
            offspring = varOr(self.population + extra, self, self.conf.lambda_, 1 - self.conf.mutpb,
                              self.conf.mutpb)

            for ind in offspring:
                # just a little extra for when we later want to analyze the hof manually
                ind.generation = gen

            for ind in self.population:
                del ind.fitness.values
            candidates = self.population + offspring

            nevals, total_steps, current_seed, candidates = self.evaluate_candidates(candidates)
            if self.hof is not None:
                self.hof.update(offspring)

            record = self.stats.compile(candidates) if self.stats is not None else {}
            self.population[:] = self.select(candidates, self.conf.mu)

            validation_fitness, validation_steps = self.validation_runs(self.population)
            self.logbook.record(gen=gen, nevals=nevals, steps=total_steps + validation_steps,
                                val_fit=validation_fitness, **record)
            if verbose:
                print(self.logbook.stream)
            self.write_checkpoint(data=dict(generation=gen, halloffame=self.hof, population=self.population,
                                            logbook=self.logbook, last_seed=current_seed, strategy=None,
                                            recorded_individuals=self.recorded_individuals))
        self.final_checkpoint_data = dict(generation=gen, halloffame=self.hof, population=self.population,
                                          logbook=self.logbook, last_seed=current_seed, strategy=None,
                                          recorded_individuals=self.recorded_individuals)
        return self.logbook

    @classmethod
    def strip_strategy_from_population(cls, population, strategy_parameter_per_gene=False):
        if len(population) == 0:
            return population
        return list(np.array(population)[:,
                    :-(cls.get_number_genes_for_strategy(len(population[0]), strategy_parameter_per_gene))])

    def get_number_genes_for_strategy(self, individual_size):
        return individual_size // 2 if self.conf.strategy_parameter_per_gene else 2

    @classmethod
    def strip_strategy_from_individual(cls, individual, strategy_parameter_per_gene=False):
        return list(
            np.array(individual)[:-(cls.get_number_genes_for_strategy(len(individual), strategy_parameter_per_gene))])
