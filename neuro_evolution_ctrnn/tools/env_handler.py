import gym
import logging
from gym.envs.algorithmic.algorithmic_env import AlgorithmicEnv
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from tools.configurations import EpisodeRunnerCfg

from gym import Wrapper
from bz2 import BZ2Compressor
from typing import Union, Iterable


class EnvHandler:
    """this class creates and modifies openAI-Environment."""

    def __init__(self, config: EpisodeRunnerCfg):
        self.conf = config

    def make_env(self, env_id: str):

        env = gym.make(env_id)

        if env_id == "Reverse-v0":
            # these options are specific to reverse-v0 and aren't important enough to be part of the
            # global configuration file.
            env.env.last = 15
            env.env.min_length = 7
            logging.info("creating env with min_length " + str(
                env.env.min_length) + " and also comparing results over the last " + str(env.env.last) + " runs.")
        if env_id.startswith("QbertNoFrameskip"):
            logging.info("wrapping env in AtariPreprocessing")
            env = AtariPreprocessing(env, screen_size=16, scale_obs=True)

        if env_id == "Reverse-v0":
            logging.info("wrapping env in ReverseWrapper")
            env = ReverseWrapper(env)

        if str(env_id).startswith("BipedalWalker"):
            logging.info("wrapping env in Box2DWalkerWrapper")
            env = Box2DWalkerWrapper(env)

        logging.info("wrapping env in BehaviorWrapper")
        env = BehaviorWrapper(env, self.conf.behavior_from_observation, self.conf.behavioral_interval,
                              self.conf.behavioral_max_length)
        return env


class BehaviorWrapper(Wrapper):
    def __init__(self, env, behavior_from_observation, behavioral_interval, behavioral_max_length):
        super().__init__(env)
        self.behavior_from_observation = behavior_from_observation
        self.behavioral_interval = behavioral_interval
        self.behavioral_max_length = behavioral_max_length
        self.compressed_behavior = b''
        self.compressor = BZ2Compressor(1)
        self.step_count = 0

    def reset(self, **kwargs):
        super(BehaviorWrapper, self).reset(**kwargs)
        self.compressed_behavior = b''
        self.compressor = BZ2Compressor(1)
        self.step_count = 0

    def step(self, action: Union[int, Iterable[int]]):
        ob, rew, done, info = super(BehaviorWrapper, self).step(action)

        if self.behavioral_interval \
                and self.step_count * self.behavioral_interval < self.behavioral_max_length \
                and self.step_count % self.behavioral_interval == 0:
            if self.behavior_from_observation:
                self.compressed_behavior += self.compressor.compress(bytearray(ob))
            else:
                self.behavior_compressed += self.compressor.compress(bytearray(action))
        return ob, rew, done, info

    def get_compressed_behavior(self):
        return self.compressed_behavior + self.compressor.flush()


class Box2DWalkerWrapper(Wrapper):
    """ simple speedup for bad agents, because some agents just stand still indefinitely and waste simulation time"""

    def __init__(self, *narg, **kwargs):
        super(Box2DWalkerWrapper, self).__init__(*narg, **kwargs)
        self.consecutive_non_movement = 0

    def reset(self, **kwargs):
        super(Box2DWalkerWrapper, self).reset(**kwargs)
        self.consecutive_non_movement = 0

    def step(self, action):
        ob, rew, done, info = super(Box2DWalkerWrapper, self).step(action)

        if ob[2] < 0.0001:
            self.consecutive_non_movement = self.consecutive_non_movement + 1
            if self.consecutive_non_movement > 50:
                done = True
                rew = rew - 100
        else:
            self.consecutive_non_movement = 0

        return ob, rew, done, info


class ReverseWrapper(Wrapper):
    """In reverse-v0 the readhead should be at a specific position when deciding which symbol to write next.
    This Wrapper adds a penalty when the head was in a wrong position, when a symbol was written"""

    def step(self, action):
        ob, rew, done, info = self.env.step(action)

        if done:
            if rew < 0:
                inp_act, out_act, pred = action
                dist = abs(len(self.unwrapped.target)-2 -self.unwrapped.read_head_position - self.unwrapped.write_head_position
                )
                if dist > 0:
                    rew -= 1. * dist
                if self.unwrapped.MOVEMENTS[inp_act] != 'left':
                    rew -= 1

        return ob, rew, done, info
