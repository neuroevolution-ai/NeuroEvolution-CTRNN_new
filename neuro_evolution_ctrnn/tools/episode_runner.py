import numpy as np
import gym

from tools.helper import set_random_seeds, output_to_action
from tools.configurations import IEpisodeRunnerCfg, StandardEpisodeRunnerCfg, MemoryExperimentCfg, IBrainCfg
from tools.dask_handler import get_current_worker
from tools.env_handler import EnvHandler
from brains.i_brain import IBrain

class IEpisodeRunner:
    def __init__(self, config: IEpisodeRunnerCfg, brain_conf: IBrainCfg, action_space, brain_class, input_space,
                 output_space, env_template):
        self.config = config
        self.brain_conf = brain_conf
        self.action_space = action_space
        self.brain_class = brain_class
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_template.spec.id

    def eval_fitness(self, individual, seed):
        pass


class EpisodeRunner(IEpisodeRunner):
    def __init__(self, config: StandardEpisodeRunnerCfg, brain_conf: IBrainCfg, action_space, brain_class, input_space,
                 output_space, env_template):
        super().__init__(config, brain_conf, action_space, brain_class, input_space, output_space, env_template)
        self.env_handler = EnvHandler(self.config)

    def eval_fitness(self, individual, seed):
        if self.config.reuse_env:
            try:
                env = get_current_worker().env
            except:
                if hasattr(self, "env"):
                    env = self.env
                else:
                    self.env = env =  self.env_handler.make_env(self.env_id)
        else:
            env = self.env_handler.make_env(self.env_id)
        set_random_seeds(seed, env)
        fitness_total = 0
        for i in range(self.config.number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual,
                                     self.brain_conf)
            ob = env.reset()
            done = False
            while not done:
                brain_output = brain.step(ob)
                action = output_to_action(brain_output, self.output_space)
                ob, rew, done, info = env.step(action)
                fitness_current += rew
            fitness_total += fitness_current

        return fitness_total / self.config.number_fitness_runs, env.get_compressed_behavior(),


class MemoryEpisodeRunner(IEpisodeRunner):
    def __init__(self, config: MemoryExperimentCfg, brain_conf: IBrainCfg, action_space, brain_class, input_space,
                 output_space, env_template):
        super().__init__(config, brain_conf, action_space, brain_class, input_space, output_space, env_template)

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
            env._max_episode_steps = self.config.observation_frames + self.config.memory_frames + self.config.action_frames

            # Create brain
            brain = self.brain_class(self.input_space, self.output_space, individual, self.brain_conf)
            output_size = IBrain._size_from_space(self.output_space)
            t = 0
            done = False
            while not done:

                # Perform step of the brain simulation
                brain_output = brain.step(ob)
                action = output_to_action(brain_output, self.output_space)

                if t <= self.config.observation_frames + self.config.memory_frames:
                    action = np.zeros(output_size)

                # Perform step of the environment simulation
                ob, rew, done, info = env.step(action)

                if t >= self.config.observation_frames:
                    for index in observation_mask:
                        ob[index] = 0.0

                if t >= self.config.observation_frames + self.config.memory_frames:
                    fitness_current += rew

                t += 1

        return fitness_current / self.config.number_fitness_runs,
