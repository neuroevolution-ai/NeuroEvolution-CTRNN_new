import gym
import pybullet_envs  # unused import is needed to register pybullet envs
import gym_memory_environments
import logging
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from tools.configurations import EpisodeRunnerCfg, ReacherMemoryEnvAttributesCfg
from tools.atari_wrappers import EpisodicLifeEnv
from gym import Wrapper
from bz2 import BZ2Compressor
from typing import Union, Iterable
import numpy as np
import cv2
from gym.spaces import Box


class EnvHandler:
    """this class creates and modifies openAI-Environment."""

    def __init__(self, config: EpisodeRunnerCfg):
        self.config = config

    def make_env(self, env_id: str):
        if env_id == "ReacherMemory-v0" or env_id == "ReacherMemoryDynamic-v0":
            assert isinstance(self.config.environment_attributes, ReacherMemoryEnvAttributesCfg), \
                "For the environment 'ReacherMemory-v0' one must provide the ReacherMemoryEnvAttributesCfg" \
                " (config.environment_attributes)"

            env = gym.make(
                env_id,
                observation_frames=self.config.environment_attributes.observation_frames,
                memory_frames=self.config.environment_attributes.memory_frames,
                action_frames=self.config.environment_attributes.action_frames)
        elif env_id.startswith("procgen"):
            logging.info("initiating procgen with memory")
            env = gym.make(env_id,
                           distribution_mode="memory",
                           use_monochrome_assets=False,
                           restrict_themes=True,
                           use_backgrounds=False)
            env = ProcEnvWrapper(env)
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

            # terminal_on_life_loss behaves different than EpisodicLifeEnv
            # terminal_on_life_loss resets the env when the first life is loss so the next agent will start fresh
            # EpisodicLifeEnv does not reset the env, so the next agent will continue where the last one died.
            # env = AtariPreprocessing(env, screen_size=32, scale_obs=True, terminal_on_life_loss=False)
            # env = EpisodicLifeEnv(env)
            env = AtariPreprocessing(env, screen_size=32, scale_obs=True, terminal_on_life_loss=True)

        if env.spec.id.startswith("Qbert"):
            logging.info("wrapping env in QbertGlitchlessWrapper")
            env = QbertGlitchlessWrapper(env)

        if env_id == "Reverse-v0":
            logging.info("wrapping env in ReverseWrapper")
            env = ReverseWrapper(env)

        if str(env_id).startswith("BipedalWalker"):
            logging.info("wrapping env in Box2DWalkerWrapper")
            env = Box2DWalkerWrapper(env)

        if self.config.novelty:
            if self.config.novelty.behavior_source in ['observation', 'action', 'state']:
                logging.info("wrapping env in BehaviorWrapper")
                env = BehaviorWrapper(env, self.config.novelty.behavior_source,
                                      self.config.novelty.behavioral_interval,
                                      self.config.novelty.behavioral_max_length)

        if self.config.max_steps_per_run:
            logging.info("wrapping env in MaxStepWrapper")
            env = MaxStepWrapper(env, max_steps=self.config.max_steps_per_run, penalty=self.config.max_steps_penalty)

        return env


class ProcEnvWrapper(Wrapper):

    def __init__(self, env):
        super(ProcEnvWrapper, self).__init__(env)
        self.screen_size = 16
        self.obs_dtype = np.float16
        self.observation_space = Box(low=0, high=1,
                                     shape=(self.screen_size, self.screen_size, 3),
                                     dtype=self.obs_dtype)

    def _transform_ob(self, ob):
        ob = cv2.resize(ob, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return np.asarray(ob, dtype=self.obs_dtype) / 255.0

    def step(self, action):
        ob, rew, done, info = super(ProcEnvWrapper, self).step(action)
        return self._transform_ob(ob), rew, done, info

    def reset(self):
        return self._transform_ob(super(ProcEnvWrapper, self).reset())


class MaxStepWrapper(Wrapper):
    def __init__(self, env, max_steps, penalty):
        super().__init__(env)
        self.steps = 0
        self.max_steps = max_steps
        self.penalty = penalty

    def reset(self, **kwargs):
        self.steps = 0
        return super(MaxStepWrapper, self).reset(**kwargs)

    def step(self, action: Union[int, Iterable[int]]):
        self.steps += 1
        ob, rew, done, info = super(MaxStepWrapper, self).step(action)
        if self.steps > self.max_steps:
            logging.info("step limit reached")
            done = True
            rew += self.penalty
        return ob, rew, done, info


class QbertGlitchlessWrapper(Wrapper):
    def step(self, action: Union[int, Iterable[int]]):
        ob, rew, done, info = super(QbertGlitchlessWrapper, self).step(action)
        if rew == 500 or rew == 525:
            logging.debug("remove reward to avoid luring enemy into abyss")
            rew = 0
        if rew == 300 or rew == 325:
            logging.debug("removed reward from fruit to avoid repetitive behavior")
            rew = 0
        return ob, rew, done, info


class BehaviorWrapper(Wrapper):
    def __init__(self, env, behavior_source, behavioral_interval, behavioral_max_length):
        super().__init__(env)
        self.behavior_source = behavior_source
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
        if self.behavior_source == "observation":
            self._record(ob)
        elif self.behavior_source == "action":
            self._record(action)
        elif self.behavior_source == "state":
            if hasattr(self.env.unwrapped, "model") and "PyMjModel" in str(type(self.env.unwrapped.model)):
                # since float16.max is only around 65500, we need to make it a little smaller
                data = np.array(self.env.unwrapped.sim.data.qpos.flat) * 10e-3
                self._record(data)
            elif self.env.spec.id.endswith("NoFrameskip-v4"):
                # this is an atari env
                # noinspection PyProtectedMember
                self._record(self.env.unwrapped._get_ram())
            else:
                raise RuntimeError('behavior_source=="state" is unsupported for this environment')
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
