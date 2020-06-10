from deap import base
from deap import creator
from deap import tools
from tools import algorithms
import numpy as np
import random
from functools import partial
from optimizer.i_optimizer import IOptimizer
from tools.configurations import OptimizerMuLambdaCfg
from typing import Callable
from tools.helper import get_checkpoint, normalized_compression_distance, euklidian_distance


def sel_elitist_tournament(individuals, mu, k_elitist, k_tournament, tournsize, fit_attr="fitness"):
    return tools.selBest(individuals, int(k_elitist * mu), fit_attr="fitness") + \
           tools.selTournament(individuals, int(k_tournament * mu), tournsize=tournsize, fit_attr="fitness")


class OptimizerMuPlusLambda(IOptimizer[OptimizerMuLambdaCfg]):
    @staticmethod
    def create_classes():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, typecode='b', fitness=creator.FitnessMax)

    def __init__(self, eval_fitness: Callable, individual_size: int, conf: OptimizerMuLambdaCfg, stats, map_func=map,
                 hof: tools.HallOfFame = tools.HallOfFame(5), from_checkoint=None):
        super(OptimizerMuPlusLambda, self).__init__(eval_fitness, individual_size, conf, stats, map_func,
                                                    hof, from_checkoint)
        self.create_classes()
        self.toolbox = toolbox = base.Toolbox()
        self.conf = conf
        self.hof = hof
        toolbox.stats = stats

        toolbox.register("map", map_func)
        toolbox.register("evaluate", eval_fitness)

        if self.conf.mutation_learned:
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
            partial(tools.cxUniform, indpb=self.conf.mate_indpb)
        ]

        mut_list = [
            partial(tools.mutGaussian,
                    mu=0.0,
                    sigma=self.conf.mutation_Gaussian_sigma_1,
                    indpb=self.conf.mutation_Gaussian_indpb_1),
            partial(tools.mutGaussian,
                    mu=0.0,
                    sigma=self.conf.mutation_Gaussian_sigma_2,
                    indpb=self.conf.mutation_Gaussian_indpb_2)
        ]

        def mate(ind1, ind2):
            return random.choice(mate_list)(ind1, ind2)

        def mutate(ind1):
            return random.choice(mut_list)(ind1)

        def fct_mutation_learned(ind1):
            # need to clip in up-direction because too large numbers create overflows
            # need to clip below to avoid stagnations, which happens when a top individuals
            # mutates bad strategy parameters
            ind1[-1] = np.clip(ind1[-1], -3, 3)
            ind1[-2] = np.clip(ind1[-2], -3, 3)

            sigma = 2 ** ind1[-1]
            indpb = 4 ** (ind1[-2] - 2)
            return tools.mutGaussian(individual=ind1, mu=0, sigma=sigma, indpb=indpb)

        toolbox.register("mate", mate)

        if self.conf.mutation_learned:
            toolbox.register("mutate", fct_mutation_learned)
        else:
            toolbox.register("mutate", mutate)
        toolbox.conf = conf
        toolbox.register("select", tools.selTournament, tournsize=self.conf.tournsize)

        self.register_checkpoints(toolbox, conf.checkpoint_frequency)

        if from_checkoint:
            cp = get_checkpoint(from_checkoint)
            toolbox.initial_generation = cp["generation"] + 1
            toolbox.initial_seed = cp["last_seed"]
            toolbox.population = cp["population"]
            toolbox.logbook = cp["logbook"]
            toolbox.recorded_individuals = cp["recorded_individuals"]
            self.hof = cp["halloffame"]
        else:
            toolbox.initial_generation = 0
            toolbox.initial_seed = None
            toolbox.population = self.toolbox.population(n=self.conf.mu + self.conf.mu_mixed + self.conf.mu_novel)
            toolbox.logbook = self.create_logbook()
            toolbox.recorded_individuals = []
            self.hof = hof

        if conf.distance == "euclid":
            toolbox.register("get_distance", euklidian_distance)
        elif conf.distance == "NCD":
            toolbox.register("get_distance", normalized_compression_distance)
        else:
            raise RuntimeError("unknown configuration value for distance: " + str(conf.distance))

    def train(self, number_generations) -> tools.Logbook:
        return algorithms.eaMuPlusLambda(toolbox=self.toolbox, ngen=number_generations, halloffame=self.hof)
