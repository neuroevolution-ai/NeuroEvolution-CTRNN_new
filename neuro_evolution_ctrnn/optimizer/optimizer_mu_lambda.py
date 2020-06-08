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
from tools.helper import get_checkpoint


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

        toolbox.register("mate", mate)
        toolbox.register("mutate", mutate)

        toolbox.register("select",
                         sel_elitist_tournament,
                         k_elitist=int(self.conf.elitist_ratio),
                         k_tournament=1.0 - int(
                             self.conf.elitist_ratio),
                         tournsize=self.conf.tournsize)
        self.register_checkpoints(toolbox, conf.checkpoint_frequency)
        toolbox.mu = int(self.conf.population_size * self.conf.mu)
        toolbox.lambda_ = int(self.conf.population_size * self.conf.lambda_)
        toolbox.cxpb = 1.0 - self.conf.mutpb
        toolbox.mutpb = self.conf.mutpb
        toolbox.novel_base = self.conf.novel_base
        toolbox.max_recorded_behaviors = self.conf.max_recorded_behaviors

        def create_seeds_for_evaluation(number_of_seeds):
            if self.conf.keep_seeds_fixed_during_generation:
                return np.ones(number_of_seeds, dtype=np.int64) * random.randint(1, 1000)
            else:
                return np.random.randint(1, 10000, size=number_of_seeds)

        toolbox.register("create_seeds_for_evaluation", create_seeds_for_evaluation)

        if from_checkoint:
            cp = get_checkpoint(from_checkoint)
            toolbox.initial_generation = cp["generation"] + 1
            toolbox.initial_seed = cp["last_seed"]
            toolbox.population = cp["population"]
            toolbox.logbook = cp["logbook"]
            toolbox.recorded_individuals = cp["recorded_individuals"]
            self.hof = cp["halloffame"]
        else:
            toolbox.logbook = tools.Logbook()
            toolbox.initial_generation = 0
            toolbox.initial_seed = None
            toolbox.population = self.toolbox.population(n=int(self.conf.population_size))
            toolbox.logbook = tools.Logbook()
            toolbox.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
            toolbox.recorded_individuals = []
            self.hof = hof

    def train(self, number_generations) -> tools.Logbook:

        return algorithms.eaMuPlusLambda(
            toolbox=self.toolbox,
            ngen=number_generations,
            halloffame=self.hof,
            include_parents_in_next_generation=self.conf.include_parents_in_next_generation
        )
