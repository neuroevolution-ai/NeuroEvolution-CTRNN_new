import numpy as np
import random
from collections import namedtuple
from neuro_evolution_ctrnn.tools.helper import set_random_seeds
import gym

EpisodeRunnerCfg = namedtuple("EpisodeRunnerCfg", [
    "number_fitness_runs", "keep_env_seed_fixed_during_generation", ])


class EpisodeRunner(object):
    def __init__(self, conf: EpisodeRunnerCfg, brain_conf: object, discrete_actions, brain_class, input_size,
                 output_size, env_template):
        self.conf = conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.brain_conf = brain_conf
        self.input_size = input_size
        self.output_size = output_size
        self.env_id = env_template.spec.id

    def eval_fitness(self, individual, seed):
        env = gym.make(self.env_id)
        set_random_seeds(seed, env)
        brain = self.brain_class(self.input_size, self.output_size, individual,
                                 self.brain_conf)

        fitness_current = 0
        for i in range(self.conf.number_fitness_runs):
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

        return fitness_current / self.conf.number_fitness_runs,
