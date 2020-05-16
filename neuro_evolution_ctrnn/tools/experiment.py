import numpy as np
import time
import gym
import json
import random
from deap import tools
from scoop import futures
from collections import namedtuple

from neuro_evolution_ctrnn.brains.continuous_time_rnn import ContrinuousTimeRNN, ContrinuousTimeRNNCfg
# import brains.layered_nn as lnn
from neuro_evolution_ctrnn.tools.episode_runner import EpisodeRunner, EpisodeRunnerCfg
from neuro_evolution_ctrnn.tools.result_handler import ResultHandler
from neuro_evolution_ctrnn.tools.trainer_cma_es import TrainerCmaEs, TrainerCmaEsCfg

# from neuro_evolution_ctrnn.tools.trainer_mu_plus_lambda import TrainerMuPlusLambda

ExperimentCfg = namedtuple("ExperimentCfg", [
    "neural_network_type", "environment", "random_seed", "random_seed_for_environment",
    "trainer_type", "number_generations", "brain", "episode_runner", "trainer"
])


class Experiment(object):

    def __init__(self, configuration_path, result_path, from_checkpoint=None):
        self.configuration_path = configuration_path
        self.result_path = result_path
        self.from_checkpoint = from_checkpoint
        self.config = self._parse_config(self.configuration_path)
        self._setup()

    def _parse_config(self, json_path):
        # Load configuration file
        with open(json_path, "r") as read_file:
            self._config_dict_raw = json.load(read_file)
        config_dict = self._config_dict_raw.copy()
        if config_dict["neural_network_type"] == 'CTRNN':
            self.brain_class = ContrinuousTimeRNN
            brain_cfg_class = ContrinuousTimeRNNCfg
        else:
            raise RuntimeError("unknown neural_network_type: " + str(config_dict["neural_network_type"]))

        if config_dict["trainer_type"] == 'CMA_ES':
            self.trainer_class = TrainerCmaEs
            trainer_cfg_class = TrainerCmaEsCfg
        else:
            raise RuntimeError("unknown trainer_type: " + str(config_dict["neural_network_type"]))

        # turned json into nested named tuples so python's type-hinting can do its magic
        # bonus: config becomes immutable
        config_dict["episode_runner"] = EpisodeRunnerCfg(**(config_dict["episode_runner"]))
        config_dict["trainer"] = trainer_cfg_class(**(config_dict["trainer"]))
        config_dict["brain"] = brain_cfg_class(**(config_dict["brain"]))
        return ExperimentCfg(**config_dict)

    def _setup(self):

        env = gym.make(self.config.environment)

        # Set random seeds
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        if self.config.random_seed_for_environment:
            env.seed(self.config.random_seed_for_environment)

        # Get individual size
        self.input_size = env.observation_space.shape[0]
        if env.action_space.shape:
            # e.g. box2d, mujoco
            self.output_size = env.action_space.shape[0]
            self.discrete_actions = False
        else:
            # e.g. lunarlander
            self.output_size = env.action_space.n
            self.discrete_actions = True

        self.individual_size = self.brain_class.get_individual_size(self.input_size, self.output_size,
                                                                    self.config.brain)

        ep_runner = EpisodeRunner(conf=self.config.episode_runner,
                                  brain_conf=self.config.brain,
                                  discrete_actions=self.discrete_actions, brain_class=self.brain_class,
                                  input_size=self.input_size, output_size=self.output_size, env=env)

        if self.config.trainer_type == "CMA_ES":
            self.trainer = self.trainer_class(map_func=futures.map, individual_size=self.individual_size,
                                              eval_fitness=ep_runner.evalFitness, conf=self.config.trainer, )
        else:
            raise RuntimeError("unknown trainer_type: " + str(self.config.trainer_type))

        self.result_handler = ResultHandler(result_path=self.result_path,
                                            neural_network_type=self.config.neural_network_type,
                                            config_raw=self._config_dict_raw)

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
            input_size=self.input_size,
            individual_size=self.individual_size)
        print("done")
