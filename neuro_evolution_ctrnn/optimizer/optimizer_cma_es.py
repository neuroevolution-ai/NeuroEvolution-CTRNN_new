from typing import Callable, Collection
from deap import tools, cma
from optimizer.i_optimizer import IOptimizer
from optimizer.creator_classes import Individual
from tools.configurations import OptimizerCmaEsCfg
from tools.helper import get_checkpoint


class OptimizerCmaEs(IOptimizer[OptimizerCmaEsCfg]):

    def __init__(self, eval_fitness: Callable, individual_size: int, random_seed: int, conf: OptimizerCmaEsCfg, stats,
                 map_func=map, from_checkpoint=None, reset_hof=False):
        super(OptimizerCmaEs, self).__init__(eval_fitness, random_seed, conf, stats, map_func,
                                             from_checkpoint, reset_hof)
        toolbox = self.toolbox

        if from_checkpoint:
            # TODO get_checkpoint(from_checkpoint) is already called in super()
            toolbox.strategy = get_checkpoint(from_checkpoint)["strategy"]
        else:
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
        toolbox = super().get_toolbox()
        verbose = __debug__
        halloffame = toolbox.hof
        gen = None
        current_seed = None
        for gen in range(toolbox.initial_generation, number_generations):
            population: Collection = toolbox.generate()
            for ind in population:
                # just a little extra for when we later want to analyze the hof manually
                ind.generation = gen
            super().record_individuals(population)
            nevals, total_steps, current_seed, canndidates = super().evaluate_candidates(population)
            population = canndidates
            if halloffame is not None:
                halloffame.update(population)
            toolbox.update(population)
            record: dict = toolbox.stats.compile(population)

            validation_fitness, validation_steps = super().validation_runs(population)
            toolbox.logbook.record(gen=gen, nevals=nevals, steps=total_steps + validation_steps,
                                   val_fit=validation_fitness, **record)
            if verbose:
                print(toolbox.logbook.stream)
            if toolbox.checkpoint:
                toolbox.checkpoint(data=dict(generation=gen, halloffame=halloffame,
                                             logbook=toolbox.logbook, last_seed=current_seed, strategy=toolbox.strategy,
                                             recorded_individuals=toolbox.recorded_individuals))

        toolbox.final_checkpoint_data = dict(generation=gen, halloffame=halloffame,
                                             logbook=toolbox.logbook, last_seed=current_seed, strategy=toolbox.strategy,
                                             recorded_individuals=toolbox.recorded_individuals)
        return toolbox.logbook
