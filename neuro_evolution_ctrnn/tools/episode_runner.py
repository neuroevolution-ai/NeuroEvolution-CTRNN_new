import numpy as np
import gym
from tools.helper import set_random_seeds
from tools.configurations import EpisodeRunnerCfg
import logging
from tools.dask_handler import get_current_worker
from typing import List, Union
from bz2 import BZ2Compressor


class EpisodeRunner(object):
    def __init__(self, conf: EpisodeRunnerCfg, brain_conf: object, discrete_actions, brain_class, input_space,
                 output_space, env_template):
        self.conf = conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.brain_conf = brain_conf
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_template.spec.id

    def eval_fitness(self, individual, seed):
        compressor = BZ2Compressor(1)
        if self.conf.reuse_env:
            env = get_current_worker().env
        else:
            env = gym.make(self.env_id)
        set_random_seeds(seed, env)
        fitness_total = 0
        behavior_compressed = b''
        for i in range(self.conf.number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual,
                                     self.brain_conf)
            ob = env.reset()
            done = False
            consecutive_non_movement = 0
            step_count = 0
            while not done:
                step_count += 1
                action = brain.step(ob)
                if self.conf.behavioral_interval \
                        and step_count * self.conf.behavioral_interval < self.conf.behavioral_max_length:
                    if step_count % self.conf.behavioral_interval == 0:
                        if self.conf.behavior_from_observation:
                            behavior_compressed += compressor.compress(bytearray(ob))
                        else:
                            behavior_compressed += compressor.compress(bytearray(action))

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
                if step_count > self.conf.max_steps_per_run:
                    done = True
            fitness_total += fitness_current

        return fitness_total / self.conf.number_fitness_runs, behavior_compressed + compressor.flush(),
