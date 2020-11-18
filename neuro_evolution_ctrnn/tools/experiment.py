import logging
import multiprocessing as mp
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
from brains.CNN_CTRNN import CnnCtrnn
from tools.episode_runner import EpisodeRunner
from tools.result_handler import ResultHandler
from optimizer.optimizer_cma_es import OptimizerCmaEs
from optimizer.optimizer_mu_lambda import OptimizerMuPlusLambda
from tools.helper import set_random_seeds, output_to_action
from tools.configurations import ExperimentCfg
from tools.dask_handler import DaskHandler
from tools.mp_handler import MPHandler
from tools.env_handler import EnvHandler
from brains.CNN_CTRNN import CnnCtrnn


# from neuro_evolution_ctrnn.tools.optimizer_mu_plus_lambda import OptimizerMuPlusLambda


class Experiment(object):

    def __init__(self, configuration: ExperimentCfg, result_path, parallel_framework,
                 number_of_workers=os.cpu_count(), from_checkpoint=None):
        self.result_path = result_path
        self.from_checkpoint = from_checkpoint
        self.config = configuration
        self.parallel_framework = parallel_framework
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
        logging.info("input space: " + str(self.input_space))
        logging.info("output space: " + str(self.output_space))
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
            logging.info("cnn_size: " + str(cnn_size) + "\tctrnn_size: " + str(ctrnn_size)+ "\tcnn_output: " + str(cnn_output_space))

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

        if self.number_of_workers <= 1:
            logging.warning("Continuing with 1 worker")
            self.number_of_workers = 1
            map_func = map
        else:
            system_cpu_count = os.cpu_count()
            if self.number_of_workers > system_cpu_count:
                logging.warning(
                    """You specified {} workers but your system supports only {} parallel processes. Continuing with """
                    """{} workers.""".format(self.number_of_workers, system_cpu_count, system_cpu_count))
                self.number_of_workers = os.cpu_count()

            if self.config.episode_runner.reuse_env:
                # TODO should this be renamed to multiprocessing instead of multithreading?
                logging.warning("Cannot reuse an environment on workers without multithreading.")

            if self.parallel_framework == "dask":
                map_func = DaskHandler.dask_map
                DaskHandler.init_dask(self.optimizer_class.create_classes, self.brain_class)
                if self.config.episode_runner.reuse_env:
                    DaskHandler.init_workers_with_env(self.config.environment, self.config.episode_runner)
            else:
                self.mp_handler = MPHandler(self.number_of_workers)
                map_func = self.mp_handler.map

        self.optimizer = self.optimizer_class(map_func=map_func,
                                              individual_size=self.individual_size,
                                              eval_fitness=self.ep_runner.eval_fitness, conf=self.config.optimizer,
                                              stats=stats, from_checkoint=self.from_checkpoint,
                                              random_seed=self.config.random_seed)

        self.result_handler = ResultHandler(result_path=self.result_path,
                                            neural_network_type=self.config.brain.type,
                                            config_raw=self.config.raw_dict)

    def cleanup(self):
        if self.number_of_workers > 1:
            if self.parallel_framework == "dask":
                DaskHandler.stop_dask()
            else:
                self.mp_handler.cleanup()

    def run(self):
        self.result_handler.check_path()
        start_time = time.time()
        log = self.optimizer.train(number_generations=self.config.number_generations)
        print("Time elapsed: %s" % (time.time() - start_time))
        self.result_handler.write_result(
            hof=self.optimizer.hof,
            log=log,
            time_elapsed=(time.time() - start_time),
            output_space=self.output_space,
            input_space=self.input_space,
            individual_size=self.individual_size)
        self.cleanup()
        print("Done")
