import numpy as np
import gym
from tools.helper import set_random_seeds, init_dask, stop_dask
from tools.configurations import EpisodeRunnerCfg
import logging
from dask.distributed import get_worker
import threading


class EpisodeRunner(object):
    def __init__(self, conf: EpisodeRunnerCfg, brain_conf: object, discrete_actions, brain_class, input_space,
                 output_size, env_template):
        self.conf = conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.brain_conf = brain_conf
        self.input_space = input_space
        self.output_size = output_size
        self.env_id = env_template.spec.id
        self.initialized = False

    def init_workers(self):
        assert threading.current_thread() is threading.main_thread()
        if self.conf.reuse_env:
            init_dask(self.env_id)
        else:
            init_dask()
        self.initialized = True

    def shutdown(self):
        assert threading.current_thread() is threading.main_thread()
        stop_dask()
        self.initialized = False

    def eval_fitness(self, individual, seed):
        assert self.initialized

        if self.conf.reuse_env:
            env = get_worker().env
        else:
            env = gym.make(self.env_id)
        set_random_seeds(seed, env)
        fitness_total = 0
        for i in range(self.conf.number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_size, individual,
                                     self.brain_conf)
            ob = env.reset()
            done = False
            consecutive_non_movement = 0
            while not done:
                action = brain.step(ob)
                if self.discrete_actions:
                    action = np.argmax(action)

                ob, rew, done, info = env.step(action)
                if str(self.env_id).startswith("BipedalWalker"):
                    # simple speedup for bad agents, because some agents just stand still indefinitely and
                    # waste simulation time
                    if ob[2] < 0.0001:
                        consecutive_non_movement = consecutive_non_movement + 1
                        if consecutive_non_movement > 50:
                            done = True
                            rew = rew - 100
                    else:
                        consecutive_non_movement = 0
                fitness_current += rew
            fitness_total += fitness_current

        return fitness_total / self.conf.number_fitness_runs,
