import numpy as np
import gym
from tools.helper import set_random_seeds
from tools.configurations import IEpisodeRunnerCfg, StandardEpisodeRunnerCfg, MemoryExperimentCfg, IBrainCfg
from tools.dask_handler import get_current_worker
from brains.i_brain import IBrain


class IEpisodeRunner:
    def __init__(self, config: IEpisodeRunnerCfg, brain_conf: IBrainCfg, discrete_actions, brain_class, input_space,
                 output_space, env_template):
        self.config = config
        self.brain_conf = brain_conf
        self.discrete_actions = discrete_actions
        self.brain_class = brain_class
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_template.spec.id

    def eval_fitness(self, individual, seed):
        pass


class EpisodeRunner(IEpisodeRunner):
    def __init__(self, config: StandardEpisodeRunnerCfg, brain_conf: IBrainCfg, discrete_actions, brain_class,
                 input_space, output_space, env_template):
        super().__init__(config, brain_conf, discrete_actions, brain_class, input_space, output_space, env_template)

    def eval_fitness(self, individual, seed):
        if self.config.reuse_env:
            env = get_current_worker().env
        else:
            env = gym.make(self.env_id)
        set_random_seeds(seed, env)
        fitness_total = 0
        for i in range(self.config.number_fitness_runs):
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

        return fitness_total / self.config.number_fitness_runs,


class MemoryEpisodeRunner(IEpisodeRunner):
    def __init__(self, config: MemoryExperimentCfg, brain_conf: IBrainCfg, discrete_actions, brain_class, input_space,
                 output_space, env_template):
        super().__init__(config, brain_conf, discrete_actions, brain_class, input_space, output_space, env_template)

    def eval_fitness(self, individual, seed):
        if self.config.reuse_env:
            env = get_current_worker().env
        else:
            env = gym.make(self.env_id)

        set_random_seeds(seed, env)
        fitness_current = 0
        observation_mask = self.config.observation_mask

        for i in range(self.config.number_fitness_runs):

            # TODO is this still necessary (increasing the seed for each worker?
            # if configuration_data["random_seed_for_environment"] is not -1:
            #     env.seed(configuration_data["random_seed_for_environment"] + i)

            ob = env.reset()
            max_episode_steps = self.config.observation_frames + self.config.memory_frames + self.config.action_frames

            # Create brain
            brain = self.brain_class(self.input_space, self.output_space, individual, self.brain_conf)
            input_size = IBrain._size_from_space(self.input_space)
            t = 0
            for _ in range(max_episode_steps):

                # Perform step of the brain simulation
                action = brain.step(ob)

                if t <= self.config.observation_frames + self.config.memory_frames:
                    action = np.zeros(input_size)

                if self.discrete_actions:
                    action = np.argmax(action)

                # Perform step of the environment simulation
                ob, rew, done, info = env.step(action)

                if t >= self.config.observation_frames:
                    for index in observation_mask:
                        ob[index] = 0.0

                if t >= self.config.observation_frames + self.config.memory_frames:
                    fitness_current += rew

                t += 1

                if done:
                    break

        return fitness_current / self.config.number_fitness_runs
