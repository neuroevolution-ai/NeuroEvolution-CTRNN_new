import random
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
        toolbox = self.toolbox

        if self.conf.strategy_parameter_per_gene:
            individual_size *= 2
        else:
            # add two genes for strategy parameters used in mutate
            individual_size += 2

        toolbox.register("indices", np.random.uniform,
                         -self.conf.initial_gene_range,
                         self.conf.initial_gene_range,
                         individual_size)
        toolbox.register("individual", tools.initIterate, Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

        toolbox.register("mate", mate)
        if self.conf.strategy_parameter_per_gene:
            toolbox.register("mutate", fct_mutation_learned_per_gene)

        else:
            toolbox.register("mutate", fct_mutation_learned)
        toolbox.register("strip_strategy_from_population", self.strip_strategy_from_population,
                         mutation_learned=True, strategy_parameter_per_gene=self.conf.strategy_parameter_per_gene)
        if self.conf.tournsize:
            toolbox.register("select", tools.selTournament, tournsize=self.conf.tournsize)
        else:
            toolbox.register("select", tools.selBest)

        if from_checkpoint:
            # TODO get_checkpoint(from_checkpoint) is already called in super()
            toolbox.population = get_checkpoint(from_checkpoint)["population"]
        else:
            toolbox.population = self.toolbox.population(n=self.conf.mu)

    def train(self, number_generations) -> tools.Logbook:
        toolbox = super().get_toolbox()
        verbose = __debug__
        include_parents_in_next_generation = True
        population = toolbox.population
        halloffame = toolbox.hof
        gen = None
        current_seed = None

        for gen in range(toolbox.initial_generation, number_generations):
            self.record_individuals(population)
            extra = []
            if halloffame.items:
                extra = list(map(toolbox.clone, random.sample(population, toolbox.conf.extra_from_hof)))
            offspring = varOr(population + extra, toolbox, toolbox.conf.lambda_, 1 - toolbox.conf.mutpb,
                              toolbox.conf.mutpb)

            for ind in offspring:
                # just a little extra for when we later want to analyze the hof manually
                ind.generation = gen

            if include_parents_in_next_generation:
                for ind in population:
                    del ind.fitness.values
                candidates = population + offspring
            else:
                candidates = offspring

            nevals, total_steps, current_seed, candidates = self.evaluate_candidates(candidates)
            if halloffame is not None:
                halloffame.update(offspring)

            record = toolbox.stats.compile(candidates) if toolbox.stats is not None else {}
            population[:] = toolbox.select(candidates, toolbox.conf.mu)
            validation_fitness, validation_steps = self.validation_runs(population)
            toolbox.logbook.record(gen=gen, nevals=nevals, steps=total_steps + validation_steps,
                                   val_fit=validation_fitness, **record)
            if verbose:
                print(toolbox.logbook.stream)
            if toolbox.checkpoint:
                toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame, population=population,
                                             logbook=toolbox.logbook, last_seed=current_seed, strategy=None,
                                             recorded_individuals=toolbox.recorded_individuals))
        toolbox.final_checkpoint_data = dict(generation=gen, halloffame=halloffame, population=population,
                                             logbook=toolbox.logbook, last_seed=current_seed, strategy=None,
                                             recorded_individuals=toolbox.recorded_individuals)
        return toolbox.logbook

    @classmethod
    def strip_strategy_from_population(cls, population, strategy_parameter_per_gene=False):
        if len(population) == 0:
            return population
        return list(np.array(population)[:, :-(cls.get_number_genes_for_strategy(len(population[0]), strategy_parameter_per_gene))])

    def get_number_genes_for_strategy(self, individual_size):
        return individual_size // 2 if self.conf.strategy_parameter_per_gene else 2

    @classmethod
    def strip_strategy_from_individual(cls, individual, strategy_parameter_per_gene=False):
        return list(
            np.array(individual)[:-(cls.get_number_genes_for_strategy(len(individual), strategy_parameter_per_gene))])
