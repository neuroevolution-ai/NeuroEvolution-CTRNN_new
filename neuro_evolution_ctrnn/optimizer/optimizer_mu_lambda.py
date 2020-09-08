from deap import base
from deap import creator
from deap import tools
from tools import algorithms
import numpy as np
import random
from optimizer.i_optimizer import IOptimizer
from tools.configurations import OptimizerMuLambdaCfg
from typing import Callable
from tools.helper import get_checkpoint


def sel_elitist_tournament(individuals, mu, k_elitist, k_tournament, tournsize, fit_attr="fitness"):
    return tools.selBest(individuals, int(k_elitist * mu), fit_attr="fitness") + \
           tools.selTournament(individuals, int(k_tournament * mu), tournsize=tournsize, fit_attr="fitness")


class OptimizerMuPlusLambda(IOptimizer[OptimizerMuLambdaCfg]):
    @staticmethod
    def create_classes():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: OptimizerMuLambdaCfg,
                 stats,
                 map_func=map, from_checkoint=None):
        super(OptimizerMuPlusLambda, self).__init__(eval_fitness, individual_size, random_seed, conf, stats, map_func,
                                                    from_checkoint)
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
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
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

        if from_checkoint:
            cp = get_checkpoint(from_checkoint)
            toolbox.initial_generation = cp["generation"] + 1
            toolbox.initial_seed = cp["last_seed"]
            toolbox.population = cp["population"]
            toolbox.logbook = cp["logbook"]
            toolbox.recorded_individuals = cp["recorded_individuals"]
            toolbox.hof = self.hof = cp["halloffame"]
        else:
            toolbox.initial_generation = 0
            toolbox.initial_seed = random_seed
            toolbox.population = self.toolbox.population(n=self.conf.mu)
            toolbox.logbook = self.create_logbook(conf)
            toolbox.recorded_individuals = []
            toolbox.hof = self.hof = tools.HallOfFame(self.conf.hof_size)

    def train(self, number_generations) -> tools.Logbook:
        return algorithms.eaMuPlusLambda(toolbox=self.toolbox, ngen=number_generations)
