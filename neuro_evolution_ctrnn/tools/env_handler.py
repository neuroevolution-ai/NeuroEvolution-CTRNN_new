import gym
import pybullet_envs
import logging
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from tools.configurations import ExperimentCfg, ReacherMemoryEnvAttributesCfg
from tools.atari_wrappers import EpisodicLifeEnv

from gym import Wrapper
from bz2 import BZ2Compressor
from typing import Union, Iterable
import numpy as np


class EnvHandler:
    """this class creates and modifies openAI-Environment."""

    def __init__(self, config: ExperimentCfg):
        self.config = config

    def make_env(self, env_id: str):
        if env_id == "ReacherMemory-v0":
            try:
                assert isinstance(self.config.environment_attributes, ReacherMemoryEnvAttributesCfg)
            except AssertionError:
                raise RuntimeError("For the environment 'ReacherMemory-v0' one must provide the"
                                   "ReacherMemoryEnvAttributesCfg (config.environment_attributes)")

            env = gym.make(
                env_id,
                observation_frames=self.config.environment_attributes.observation_frames,
                memory_frames=self.config.environment_attributes.memory_frames,
                action_frames=self.config.environment_attributes.action_frames,
                observation_mask=self.config.environment_attributes.observation_mask)
        else:
            env = gym.make(env_id)

        if env_id == "Reverse-v0":
            # these options are specific to reverse-v0 and aren't important enough to be part of the
            # global configuration file.
            env.env.last = 15
            env.env.min_length = 7
            logging.info("creating env with min_length " + str(
                env.env.min_length) + " and also comparing results over the last " + str(env.env.last) + " runs.")

        if env.spec.id.endswith("NoFrameskip-v4"):
            logging.info("wrapping env in AtariPreprocessing")
            env = AtariPreprocessing(env, screen_size=16, scale_obs=True)

            logging.info("wrapping env in EpisodicLifeEnv")
            env = EpisodicLifeEnv(env)

        if env.spec.id.startswith("Qbert"):
            logging.info("wrapping env in QbertGlitchlessWrapper")
            env = QbertGlitchlessWrapper(env)

        if env_id == "Reverse-v0":
            logging.info("wrapping env in ReverseWrapper")
            env = ReverseWrapper(env)

        if str(env_id).startswith("BipedalWalker"):
            logging.info("wrapping env in Box2DWalkerWrapper")
            env = Box2DWalkerWrapper(env)

        if self.config.episode_runner.novelty:
            logging.info("wrapping env in BehaviorWrapper")
            env = BehaviorWrapper(env,
                                  self.config.episode_runner.novelty.behavior_from_observation,
                                  self.config.episode_runner.novelty.behavioral_interval,
                                  self.config.episode_runner.novelty.behavioral_max_length)
        return env


class QbertGlitchlessWrapper(Wrapper):
    def step(self, action: Union[int, Iterable[int]]):
        ob, rew, done, info = super(QbertGlitchlessWrapper, self).step(action)
        if rew == 500:
            logging.info("QbertGlitchlessWrapper removed reward to avoid glitch abuse")

            rew = 0
        return ob, rew, done, info


class BehaviorWrapper(Wrapper):
    def __init__(self, env, behavior_from_observation, behavioral_interval, behavioral_max_length):
        super().__init__(env)
        self.behavior_from_observation = behavior_from_observation
        self.behavioral_interval = behavioral_interval
        self.behavioral_max_length = behavioral_max_length
        self.compressed_behavior = b''
        self.compressor = BZ2Compressor(1)
        self.step_count = 0
        self.aggregate = None

    def reset(self, **kwargs):
        self.compressed_behavior = b''
        self.compressor = BZ2Compressor(2)
        self.step_count = 0
        self.aggregate = None
        return super(BehaviorWrapper, self).reset(**kwargs)

    def _record(self, data):
        if self.aggregate is None:
            self.aggregate = np.array(data, dtype=np.float32)
            self.aggregate.fill(0)

        if self.behavioral_interval != 0:
            self.aggregate += np.array(data) / self.behavioral_interval

        if self.step_count * self.behavioral_interval < self.behavioral_max_length:
            if self.step_count % self.behavioral_interval == 0:
                data_bytes = np.array(self.aggregate).astype(np.float16).tobytes()
                self.compressed_behavior += self.compressor.compress(data_bytes)
                self.aggregate.fill(0)

    def step(self, action: Union[int, Iterable[int]]):
        ob, rew, done, info = super(BehaviorWrapper, self).step(action)

        if hasattr(self.env.unwrapped, "model") and "PyMjModel" in str(type(self.env.unwrapped.model)):

            # since float16.max is only around 65500, we need to make it a little smaller
            data = np.array(self.env.unwrapped.sim.data.qpos.flat) * 10e-3
            self._record(data)
        elif self.env.spec.id.endswith("NoFrameskip-v4"):
            # this is an atari env
            # noinspection PyProtectedMember
            self._record(self.env.unwrapped._get_ram())
        elif self.behavior_from_observation:
            self._record(ob)
        else:
            self._record(action)
        return ob, rew, done, info

    def get_compressed_behavior(self):
        return self.compressed_behavior + self.compressor.flush()


class Box2DWalkerWrapper(Wrapper):
    """ simple speedup for bad agents, because some agents just stand still indefinitely and waste simulation time"""

    def __init__(self, *narg, **kwargs):
        super(Box2DWalkerWrapper, self).__init__(*narg, **kwargs)
        self.consecutive_non_movement = 0

    def reset(self, **kwargs):
        self.consecutive_non_movement = 0
        return super(Box2DWalkerWrapper, self).reset(**kwargs)

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
                dist = abs(len(self.unwrapped.target)
                           - self.unwrapped.read_head_position
                           - self.unwrapped.write_head_position)
                if dist > 0:
                    rew -= 1. * dist
                if self.unwrapped.MOVEMENTS[inp_act] != 'left':
                    rew -= 1

        return ob, rew, done, info
