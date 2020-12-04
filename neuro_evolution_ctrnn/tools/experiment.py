import logging
import os
import time
from typing import Type

import numpy as np
from deap import tools

from brains.continuous_time_rnn import ContinuousTimeRNN
from brains.ffnn import FeedForwardNumPy, FeedForwardPyTorch
from brains.i_brain import IBrain
from optimizer.i_optimizer import IOptimizer
from brains.lstm import LSTMPyTorch, LSTMNumPy
from brains.concatenated_brains import ConcatenatedLSTM
from tools.episode_runner import EpisodeRunner
from tools.result_handler import ResultHandler
from optimizer.optimizer_cma_es import OptimizerCmaEs
from optimizer.optimizer_mu_lambda import OptimizerMuPlusLambda
from tools.helper import set_random_seeds
from tools.configurations import ExperimentCfg
from processing_handlers.dask_handler import DaskHandler
from processing_handlers.mp_handler import MPHandler
from processing_handlers.sequential_handler import SequentialHandler
from tools.env_handler import EnvHandler
from brains.CNN_CTRNN import CnnCtrnn


class Experiment(object):

    def __init__(self, configuration: ExperimentCfg, result_path, processing_framework,
                 number_of_workers=os.cpu_count(), from_checkpoint=None):
        self.result_path = result_path
        self.from_checkpoint = from_checkpoint
        self.config = configuration
        self.processing_framework = processing_framework
        self.number_of_workers: int = number_of_workers
        self.brain_class: Type[IBrain]
        if self.config.brain.type == "CTRNN":
            self.brain_class = ContinuousTimeRNN
        elif self.config.brain.type == "FeedForward_NumPy":
            self.brain_class = FeedForwardNumPy
        elif self.config.brain.type == "FeedForward_PyTorch":
            self.brain_class = FeedForwardPyTorch
        elif self.config.brain.type == "LSTM_PyTorch":
            self.brain_class = LSTMPyTorch
        elif self.config.brain.type == "LSTM_NumPy":
            self.brain_class = LSTMNumPy
        elif self.config.brain.type == "ConcatenatedBrain_LSTM":
            self.brain_class = ConcatenatedLSTM
        elif self.config.brain.type == "CNN_CTRNN":
            self.brain_class = CnnCtrnn
        else:
            raise RuntimeError("Unknown neural network type (config.brain.type): " + str(self.config.brain.type))

        self.optimizer_class: Type[IOptimizer]
        if self.config.optimizer.type == "CMA_ES":
            self.optimizer_class = OptimizerCmaEs
        elif self.config.optimizer.type == "MU_ES":
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
        logging.info("Input space: " + str(self.input_space))
        logging.info("Output space: " + str(self.output_space))
        self.brain_class.set_masks_globally(config=self.config.brain,
                                            input_space=self.input_space,
                                            output_space=self.output_space)

        self.individual_size = self.brain_class.get_individual_size(self.config.brain,
                                                                    input_space=self.input_space,
                                                                    output_space=self.output_space)
        logging.info("Individual size for this experiment: " + str(self.individual_size))
        if issubclass(self.brain_class, CnnCtrnn):
            cnn_size, ctrnn_size, cnn_output_space = self.brain_class._get_sub_individual_size(self.config.brain,
                                                                             input_space=self.input_space,
                                                                             output_space=self.output_space)
            logging.info("cnn_size: " + str(cnn_size) + "\tctrnn_size: " + str(ctrnn_size)+ "\tcnn_output: " +
                         str(cnn_output_space))

        self.ep_runner = EpisodeRunner(config=self.config.episode_runner, brain_config=self.config.brain,
                                       brain_class=self.brain_class, input_space=self.input_space,
                                       output_space=self.output_space, env_template=env)

        stats_fit = tools.Statistics(key=lambda ind: ind.fitness_orig)
        if self.config.episode_runner.novelty:
            stats_novel = tools.Statistics(key=lambda ind: ind.novelty)
            stats = tools.MultiStatistics(fitness=stats_fit, novelty=stats_novel)
        else:
            stats = tools.MultiStatistics(fitness=stats_fit)

        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        system_cpu_count = os.cpu_count()
        if self.number_of_workers <= 0 or self.number_of_workers > system_cpu_count:
            raise RuntimeError(
                "{} is an incorrect number of processes. Your system only supports {} workers and it must be at least "
                "1.".format(self.number_of_workers, system_cpu_count))

        if self.processing_framework == "dask":
            self.processing_handler = DaskHandler(self.number_of_workers, self.optimizer_class.create_classes,
                                                  self.brain_class, self.number_of_workers)
        elif self.processing_framework == "mp":
            self.processing_handler = MPHandler(self.number_of_workers)
        elif self.processing_framework == "sequential":
            self.processing_handler = SequentialHandler(self.number_of_workers)
        else:
            raise RuntimeError(
                "The processing framework '{}' is not supported.".format(self.processing_framework))

        map_func = self.processing_handler.map

        self.optimizer = self.optimizer_class(map_func=map_func,
                                              individual_size=self.individual_size,
                                              eval_fitness=self.ep_runner.eval_fitness, conf=self.config.optimizer,
                                              stats=stats, from_checkoint=self.from_checkpoint,
                                              random_seed=self.config.random_seed)

        self.result_handler = ResultHandler(result_path=self.result_path,
                                            neural_network_type=self.config.brain.type,
                                            config_raw=self.config.raw_dict)

    def run(self):
        self.result_handler.check_path()
        start_time = time.time()
        self.processing_handler.init_framework()
        log = self.optimizer.train(number_generations=self.config.number_generations)
        print("Time elapsed: %s" % (time.time() - start_time))
        self.result_handler.write_result(
            hof=self.optimizer.hof,
            log=log,
            time_elapsed=(time.time() - start_time),
            output_space=self.output_space,
            input_space=self.input_space,
            individual_size=self.individual_size)
        self.processing_handler.cleanup_framework()
        print("Done")
