import numpy as np
import random
from collections import namedtuple

EpisodeRunnerCfg = namedtuple("EpisodeRunnerCfg", [
    "number_fitness_runs", "keep_env_seed_fixed_during_generation", ])


class EpisodeRunner(object):
    # episode Runner can't be defined in a submodule, because
    # when it is, scoop throws lots of errors on process-end.

    def __init__(self, conf: EpisodeRunnerCfg, brain_conf: object, discrete_actions, brain_class, input_size, output_size, env):
        self.conf = conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.brain_conf = brain_conf
        self.input_size = input_size
        self.output_size = output_size
        self.env = env

    def evalFitness(self, individual):
        env = self.env
        env_seed = None
        brain = self.brain_class(self.input_size, self.output_size, individual,
                                 self.brain_conf)
        fitness_current = 0
        number_fitness_runs = self.conf.number_fitness_runs
        if env_seed:
            env.seed(env_seed)
            env.action_space.seed(env_seed)

        for i in range(number_fitness_runs):
            ob = self.env.reset()
            done = False
            consecutive_non_movement = 0
            it = 0
            while not done:
                # Perform step of the brain simulation
                action = brain.step(ob)

                if self.discrete_actions:
                    action = np.argmax(action)
                # Perform step of the environment simulation
                ob, rew, done, info = self.env.step(action)
                if str(self.env).startswith("BipedalWalker"):
                    # simple speedup for bad agents
                    if ob[2] < 0.0001:
                        consecutive_non_movement = consecutive_non_movement + 1
                        if consecutive_non_movement > 50:
                            done = True
                            rew = rew - 100
                    else:
                        consecutive_non_movement = 0
                fitness_current += rew
        return fitness_current / number_fitness_runs,
