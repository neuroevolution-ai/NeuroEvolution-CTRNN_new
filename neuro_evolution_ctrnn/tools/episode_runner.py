import numpy as np
import gym
from tools.helper import set_random_seeds
from tools.configurations import IEpisodeRunnerCfg, StandardEpisodeRunnerCfg, MemoryExperimentCfg, IBrainCfg
import logging
from tools.dask_handler import get_current_worker


class IEpisodeRunner:
    def __init__(self, conf: IEpisodeRunnerCfg, brain_conf: IBrainCfg, discrete_actions, brain_class, input_space,
                 output_space, env_template):
        self.conf = conf
        self.brain_conf = brain_conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_template.spec.id

    def eval_fitness(self, individual, seed):
        pass


class EpisodeRunner(IEpisodeRunner):
    def __init__(self, conf: StandardEpisodeRunnerCfg, brain_conf: IBrainCfg, discrete_actions, brain_class,
                 input_space, output_space, env_template):
        super().__init__(conf, brain_conf, discrete_actions, brain_class, input_space, output_space, env_template)

    def eval_fitness(self, individual, seed):
        if self.conf.reuse_env:
            env = get_current_worker().env
        else:
            env = gym.make(self.env_id)
        set_random_seeds(seed, env)
        fitness_total = 0
        for i in range(self.conf.number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual,
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


class MemoryEpisodeRunner(IEpisodeRunner):
    def __init__(self, conf: MemoryExperimentCfg, brain_conf: IBrainCfg, discrete_actions, brain_class, input_space,
                 output_space, env_template):
        super().__init__(conf, brain_conf, discrete_actions, brain_class, input_space, output_space, env_template)

    def eval_fitness(self, individual, seed):
        if self.conf.reuse_env:
            env = get_current_worker().env
        else:
            env = gym.make(self.env_id)

        set_random_seeds(seed, env)


        print("Observation frames {}".format(self.conf.observation_frames))
        print("Memory frames {}".format(self.conf.memory_frames))
        print("Action frames {}".format(self.conf.action_frames))
