import numpy as np
import time
import gym
from deap import tools
import logging
from typing import Type

from brains.continuous_time_rnn import ContinuousTimeRNN
from brains.layered_nn import LayeredNN
from brains.i_brain import IBrain
from optimizer.i_optimizer import IOptimizer
from brains.lstm import LSTMPyTorch, LSTMNumPy
# import brains.layered_nn as lnn
from tools.episode_runner import EpisodeRunner
from tools.result_handler import ResultHandler
from optimizer.optimizer_cma_es import OptimizerCmaEs
from optimizer.optimizer_mu_lambda import OptimizerMuPlusLambda
from tools.helper import set_random_seeds, output_to_action
from tools.configurations import ExperimentCfg
from tools.dask_handler import DaskHandler
from tools.env_handler import EnvHandler


# from neuro_evolution_ctrnn.tools.optimizer_mu_plus_lambda import OptimizerMuPlusLambda


class Experiment(object):

    def __init__(self, configuration: ExperimentCfg, result_path, from_checkpoint=None):
        self.result_path = result_path
        self.from_checkpoint = from_checkpoint
        self.config = configuration
        self.brain_class: Type[IBrain]
        if self.config.brain.type == "CTRNN":
            self.brain_class = ContinuousTimeRNN
        elif self.config.brain.type == "LNN":
            self.brain_class = LayeredNN
        elif self.config.brain.type == "LSTM_PyTorch":
            self.brain_class = LSTMPyTorch
        elif self.config.brain.type == "LSTM_NumPy":
            self.brain_class = LSTMNumPy
        else:
            raise RuntimeError("Unknown neural network type (config.brain.type): " + str(self.config.brain.type))

        self.optimizer_class: Type[IOptimizer]
        if self.config.optimizer.type == 'CMA_ES':
            self.optimizer_class = OptimizerCmaEs
        elif self.config.optimizer.type == 'MU_ES':
            self.optimizer_class = OptimizerMuPlusLambda
        else:
            raise RuntimeError("Unknown optimizer (config.optimizer.type): " + str(self.config.optimizer.type))

        self._setup()

    def _setup(self):
        env_handler = EnvHandler(self.config.episode_runner)
        env = env_handler.make_env(self.config.environment)
        # note: the environment defined here is only used to initialize other classes, but the
        # actual simulation will happen on freshly created local  environments on the episode runners
        # to avoid concurrency problems that would arise from a shared global state
        self.env_template = env
        set_random_seeds(self.config.random_seed, env)
        self.input_space = env.observation_space
        self.output_space = env.action_space

        self.brain_class.set_masks_globally(config=self.config.brain,
                                            input_space=self.input_space,
                                            output_space=self.output_space)

        self.individual_size = self.brain_class.get_individual_size(self.config.brain,
                                                                    input_space=self.input_space,
                                                                    output_space=self.output_space)
        logging.info("Individual size for this experiment: " + str(self.individual_size))

        self.ep_runner = EpisodeRunner(conf=self.config.episode_runner,
                                       brain_conf=self.config.brain,
                                       action_space=self.output_space, brain_class=self.brain_class,
                                       input_space=self.input_space, output_space=self.output_space, env_template=env)

        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
        if self.config.episode_runner.novelty:
            stats_novel = tools.Statistics(key=lambda ind: ind.novelty)
            stats = tools.MultiStatistics(fitness=stats_fit, novelty=stats_novel)
        else:
            stats = tools.MultiStatistics(fitness=stats_fit)

        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)
        if self.config.use_worker_processes:
            map_func = DaskHandler.dask_map
        else:
            map_func = map
            if self.config.episode_runner.reuse_env:
                logging.warning("can't reuse env on workers without multithreading. ")

        self.optimizer = self.optimizer_class(map_func=map_func,
                                              individual_size=self.individual_size,
                                              eval_fitness=self.ep_runner.eval_fitness, conf=self.config.optimizer,
                                              stats=stats, from_checkoint=self.from_checkpoint)

        self.result_handler = ResultHandler(result_path=self.result_path,
                                            neural_network_type=self.config.brain.type,
                                            config_raw=self.config.raw_dict)

    def run(self):
        self.result_handler.check_path()
        start_time = time.time()

        DaskHandler.init_dask(self.optimizer_class.create_classes, self.brain_class)
        if self.config.episode_runner.reuse_env and self.config.use_worker_processes:
            DaskHandler.init_workers_with_env(self.env_template.spec.id, self.config.episode_runner)
        log = self.optimizer.train(number_generations=self.config.number_generations)
        print("Time elapsed: %s" % (time.time() - start_time))
        self.result_handler.write_result(
            hof=self.optimizer.hof,
            log=log,
            time_elapsed=(time.time() - start_time),
            output_space=self.output_space,
            input_space=self.input_space,
            individual_size=self.individual_size)
        DaskHandler.stop_dask()
        print("Done")

    def visualize(self, individuals, brain_vis_handler, rounds_per_individual=1, neuron_vis=False, slow_down=0):
        env_handler = EnvHandler(self.config.episode_runner)
        env = env_handler.make_env(self.config.environment)
        env.render()
        if hasattr(self.config.optimizer, "mutation_learned"):
            # sometimes there are also optimizing strategies encoded in the genome. These parameters
            # are not part of the brain and need to be removed from the genome before initializing the brain.
            individuals = self.optimizer.strip_strategy_from_population(individuals,
                                                                        self.config.optimizer.mutation_learned)

        for individual in individuals:
            set_random_seeds(self.config.random_seed, env)

            brain = self.brain_class(input_space=self.input_space,
                                     output_space=self.output_space,
                                     individual=individual,
                                     config=self.config.brain)

            for i in range(rounds_per_individual):
                fitness_current = 0
                ob = env.reset()
                done = False
                if neuron_vis:
                    brain_vis = brain_vis_handler.launch_new_visualization(brain)
                else:
                    brain_vis = None
                step_count = 0
                while not done:
                    step_count += 1
                    action = brain.step(ob)
                    if brain_vis:
                        brain_vis.process_update(in_values=ob, out_values=action)
                    action = output_to_action(action, self.output_space)
                    ob, rew, done, info = env.step(action)
                    if slow_down:
                        time.sleep(slow_down / 1000.0)
                    fitness_current += rew
                    env.render()
                print("steps: " + str(step_count) + " \tReward: " + str(fitness_current))
