import numpy as np
import time
import gym
from deap import tools
from scoop import futures
import logging

from brains.continuous_time_rnn import ContinuousTimeRNN
from brains.layered_nn import LayeredNN
from brains.i_brain import IBrain
# import brains.layered_nn as lnn
from tools.episode_runner import EpisodeRunner
from tools.result_handler import ResultHandler
from tools.optimizer_cma_es import OptimizerCmaEs
from tools.helper import set_random_seeds
from tools.configurations import ExperimentCfg
from tools.dask_handler import DaskHandler


# from neuro_evolution_ctrnn.tools.optimizer_mu_plus_lambda import OptimizerMuPlusLambda


class Experiment(object):

    def __init__(self, configuration: ExperimentCfg, result_path, from_checkpoint=None):
        self.result_path = result_path
        self.from_checkpoint = from_checkpoint
        self.config = configuration
        self.brain_class: IBrain
        if self.config.brain.type == 'CTRNN':
            self.brain_class = ContinuousTimeRNN
        elif self.config.brain.type == 'LNN':
            self.brain_class = LayeredNN
        else:
            raise RuntimeError("unknown neural_network_type: " + str(self.config.brain.type))

        if self.config.optimizer.type == 'CMA_ES':
            self.optimizer_class = OptimizerCmaEs
        else:
            raise RuntimeError("unknown optimizer.type: " + str(self.config.optimizer.type))

        self._setup()

    def _setup(self):
        env = gym.make(self.config.environment)
        # note: the environment defined here is only used to initialize other classes, but the
        # actual simulation will happen on freshly created local  environments on the episode runners
        # to avoid concurrency problems that would arise from a shared global state
        self.env_template = env
        set_random_seeds(self.config.random_seed, env)
        self.input_space = env.observation_space
        self.output_space = env.action_space
        if env.action_space.shape:
            # e.g. box2d, mujoco
            self.output_size = env.action_space.shape[0]
            self.discrete_actions = False
        else:
            # e.g. lunarlander
            self.output_size = env.action_space.n
            self.discrete_actions = True

        self.brain_class.set_masks_globally(config=self.config.brain,
                                            input_space=self.input_space,
                                            output_space=self.output_space)

        self.individual_size = self.brain_class.get_individual_size(self.config.brain,
                                            input_space=self.input_space,
                                            output_space=self.output_space)
        logging.info("Infividual Size for this Experiment: " + str(self.individual_size))

        self.ep_runner = EpisodeRunner(conf=self.config.episode_runner,
                                       brain_conf=self.config.brain,
                                       discrete_actions=self.discrete_actions, brain_class=self.brain_class,
                                       input_space=self.input_space, output_space=self.output_space, env_template=env)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        if self.config.optimizer.type == "CMA_ES":
            self.optimizer = self.optimizer_class(map_func=DaskHandler.dask_map,
                                                  individual_size=self.individual_size,
                                                  eval_fitness=self.ep_runner.eval_fitness, conf=self.config.optimizer,
                                                  stats=stats, from_checkoint=self.from_checkpoint)
        else:
            raise RuntimeError("unknown optimizer.type: " + str(self.config.optimizer.type))

        self.result_handler = ResultHandler(result_path=self.result_path,
                                            neural_network_type=self.config.brain.type,
                                            config_raw=self.config.raw_dict)

    def run(self):
        self.result_handler.check_path()
        start_time = time.time()

        DaskHandler.init_dask(self.optimizer_class.create_classes, self.brain_class)
        if self.config.episode_runner.reuse_env:
            DaskHandler.init_workers_with_env(self.env_template.spec.id)
        log = self.optimizer.train(number_generations=self.config.number_generations)
        print("Time elapsed: %s" % (time.time() - start_time))
        self.result_handler.write_result(
            hof=self.optimizer.hof,
            log=log,
            time_elapsed=(time.time() - start_time),
            output_size=self.output_size,
            input_space=self.input_space,
            individual_size=self.individual_size)
        DaskHandler.stop_dask()
        print("done")

    def visualize(self, individuals, brain_vis_handler, rounds_per_individual=1, neuron_vis=False, slow_down=0):
        env = gym.make(self.config.environment)
        env.render()

        for individual in individuals:
            set_random_seeds(self.config.random_seed, env)
            brain = self.brain_class(input_space=self.input_space,
                                     output_space=self.output_space,
                                     individual=individual,
                                     config=self.config.brain)

            brain_vis = brain_vis_handler.launch_new_visualization(brain)
            for i in range(rounds_per_individual):
                fitness_current = 0
                ob = env.reset()
                done = False
                if neuron_vis:
                    brain_vis = brain_vis_handler.launch_new_visualization(brain)

                while not done:
                    action = brain.step(ob)
                    if neuron_vis:
                        brain_vis.process_update(in_values=ob, out_values=action)
                    if self.discrete_actions:
                        action = np.argmax(action)
                    ob, rew, done, info = env.step(action)
                    if slow_down:
                        time.sleep(slow_down / 1000.0)
                    fitness_current += rew
                    env.render()
                print(fitness_current)
