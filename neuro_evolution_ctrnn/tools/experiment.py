import numpy as np
import time
import gym
import json
import random
from deap import tools
from scoop import futures
from collections import namedtuple

from brains.continuous_time_rnn import ContinuousTimeRNN
# import brains.layered_nn as lnn
from tools.episode_runner import EpisodeRunner
from tools.result_handler import ResultHandler
from tools.trainer_cma_es import TrainerCmaEs
from tools.helper import set_random_seeds
from tools.configurations import ExperimentCfg


# from neuro_evolution_ctrnn.tools.trainer_mu_plus_lambda import TrainerMuPlusLambda


class Experiment(object):

    def __init__(self, configuration: ExperimentCfg, result_path, from_checkpoint=None):
        self.result_path = result_path
        self.from_checkpoint = from_checkpoint
        self.config = configuration

        if self.config.neural_network_type == 'CTRNN':
            self.brain_class = ContinuousTimeRNN
        else:
            raise RuntimeError("unknown neural_network_type: " + str(self.config.neural_network_type))

        if self.config.trainer_type == 'CMA_ES':
            self.trainer_class = TrainerCmaEs
        else:
            raise RuntimeError("unknown trainer_type: " + str(self.config.trainer_type))

        self._setup()

    def _setup(self):
        env = gym.make(self.config.environment)
        # note: the environment defined here is only used to initialize other classes, but the
        # actual simulation will happen on freshly created local  environments on the episode runners
        # to avoid concurrency problems that would arise from a shared global state
        self.env_template = env
        set_random_seeds(self.config.random_seed, env)

        # Get individual size
        self.input_space = env.observation_space.shape
        if env.action_space.shape:
            # e.g. box2d, mujoco
            self.output_size = env.action_space.shape[0]
            self.discrete_actions = False
        else:
            # e.g. lunarlander
            self.output_size = env.action_space.n
            self.discrete_actions = True

        self.individual_size = self.brain_class.get_individual_size(self.input_space, self.output_size,
                                                                    self.config.brain)

        ep_runner = EpisodeRunner(conf=self.config.episode_runner,
                                  brain_conf=self.config.brain,
                                  discrete_actions=self.discrete_actions, brain_class=self.brain_class,
                                  input_space=self.input_space, output_size=self.output_size, env_template=env)

        if self.config.trainer_type == "CMA_ES":
            self.trainer = self.trainer_class(map_func=futures.map, individual_size=self.individual_size,
                                              eval_fitness=ep_runner.eval_fitness, conf=self.config.trainer, )
        else:
            raise RuntimeError("unknown trainer_type: " + str(self.config.trainer_type))

        self.result_handler = ResultHandler(result_path=self.result_path,
                                            neural_network_type=self.config.neural_network_type,
                                            config_raw=self.config.raw_dict)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def run(self):
        self.result_handler.check_path()
        start_time = time.time()
        pop, log = self.trainer.train(self.stats, number_generations=self.config.number_generations)

        # print elapsed time
        print("Time elapsed: %s" % (time.time() - start_time))
        self.result_handler.write_result(
            hof=self.trainer.hof,
            log=log,
            time_elapsed=(time.time() - start_time),
            output_size=self.output_size,
            input_space=self.input_space,
            individual_size=self.individual_size)
        print("done")

    def visualize(self, individuals, brain_vis_handler):
        env = gym.make(self.config.environment)

        # Get individual size
        input_space = env.observation_space.shape
        output_size = env.action_space.shape[0]

        env.render()

        for individual in individuals:
            fitness_current = 0
            set_random_seeds(self.config.random_seed, env)
            ob = env.reset()
            done = False
            brain = self.brain_class(input_space=input_space,
                                     output_size=output_size,
                                     individual=individual,
                                     config=self.config.brain)
            brain_vis = brain_vis_handler.launch_new_visualization(brain)

            while not done:
                action = brain.step(ob)
                brain_vis.process_update(input=ob, output=action)
                ob, rew, done, info = env.step(action)
                fitness_current += rew
                env.render()
            print(fitness_current)
