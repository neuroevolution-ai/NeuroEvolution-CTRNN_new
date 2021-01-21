from bz2 import BZ2Compressor
import copy
import logging
from typing import Union, Iterable

import cv2
import gym
from gym.spaces import Box
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym import Wrapper
import numpy as np
import pybullet_envs  # unused import is needed to register pybullet envs

import gym_memory_environments
from tools.configurations import EpisodeRunnerCfg, ReacherMemoryEnvAttributesCfg, AtariEnvAttributesCfg
from tools.atari_wrappers import EpisodicLifeEnv
from tools.ae_wrapper import AEWrapper


class EnvHandler:
    """This class creates and modifies OpenAI-Gym environments."""

    def __init__(self, config: EpisodeRunnerCfg):
        self.config = config

    def make_env(self, env_id: str, render: bool = False, record: str = None, record_force: bool = False):
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
            logging.debug("initiating procgen with memory")
            env = ProcEnvHandler(env_id, render)
        elif env_id == 'QbertHard-v0':
            logging.debug("wrapping QbertNoFrameskip-v4 in QbertGlitchlessWrapper")
            env = QbertGlitchlessWrapper(gym.make('QbertNoFrameskip-v4'))
        elif env_id == 'ReverseShaped-v0':
            env = gym.make('Reverse-v0')
            # these options are specific to reverse-v0 and aren't important enough to be part of the
            # global configuration file.
            env.env.last = 15
            env.env.min_length = 7
            logging.debug("creating env with min_length " + str(
                env.env.min_length) + " and also comparing results over the last " + str(env.env.last) + " runs.")

            logging.debug("wrapping env in ReverseWrapper")
            env = ReverseWrapper(env)
        else:
            env = gym.make(env_id)

        if self.config.use_autoencoder:
            logging.debug("wrapping env in AEWrapper")
            env = AEWrapper(env)
        else:
            if env.spec.id.endswith("NoFrameskip-v4"):
                logging.debug("wrapping env in AtariPreprocessing")

                assert isinstance(self.config.environment_attributes, AtariEnvAttributesCfg), \
                    "For atari environment one must provide the AtariEnvAttributesCfg" \
                    " (config.environment_attributes)"

                # terminal_on_life_loss behaves different than EpisodicLifeEnv
                # terminal_on_life_loss resets the env when the first life is loss so the next agent will start fresh
                # EpisodicLifeEnv does not reset the env, so the next agent will continue where the last one died.
                # env = AtariPreprocessing(env, screen_size=32, scale_obs=True, terminal_on_life_loss=False)
                # env = EpisodicLifeEnv(env)
                env = AtariPreprocessing(env,
                                         screen_size=self.config.environment_attributes.screen_size,
                                         scale_obs=self.config.environment_attributes.scale_obs,
                                         terminal_on_life_loss=self.config.environment_attributes.terminal_on_life_loss,
                                         grayscale_obs=self.config.environment_attributes.grayscale_obs)

        if str(env_id).startswith("BipedalWalker"):
            logging.debug("wrapping env in Box2DWalkerWrapper")
            env = Box2DWalkerWrapper(env)

        if self.config.novelty:
            if self.config.novelty.behavior_source in ['observation', 'action', 'state']:
                logging.debug("wrapping env in BehaviorWrapper")
                env = BehaviorWrapper(env, self.config.novelty.behavior_source,
                                      self.config.novelty.behavioral_interval,
                                      self.config.novelty.behavioral_max_length)

        if self.config.max_steps_per_run:
            logging.debug("wrapping env in MaxStepWrapper")
            env = MaxStepWrapper(env, max_steps=self.config.max_steps_per_run, penalty=self.config.max_steps_penalty)

        if record is not None:
            env = gym.wrappers.Monitor(env, record, force=record_force)

        return env


class ProcEnvHandler(gym.Env):
    """
    This Wrapper scales to observation to values between 0 and 1.
    Additionally it implements a seed method because for reasons unknown it not implemented upstream
    """

    def __init__(self, env_id, render):
        # todo: maybe add env specific configuration, but only after issue #20 has been implemented
        self.env_id = env_id
        self.render_mode = None
        if render:
            self.render_mode = "rgb_array"
        super().__init__()
        self._env = self._make_inner_env(start_level=0)
        self.spec = copy.deepcopy(self._env.spec)  # deep copy to avoid references to inner gym
        self.action_space = self._env.action_space  # use reference, so action_space.seed() works as expected
        self.obs_dtype = np.float16
        self.input_high = 255
        self.current_level = 0
        assert self.input_high == self._env.observation_space.high.min(), "unexpected bounds for input space"
        assert self.input_high == self._env.observation_space.high.max(), "unexpected bounds for input space"
        assert 0 == self._env.observation_space.low.min(), "unexpected bounds for input space"
        assert 0 == self._env.observation_space.low.max(), "unexpected bounds for input space"
        self.observation_space = Box(low=0, high=1,
                                     shape=self._env.observation_space.shape,
                                     dtype=self.obs_dtype)

    def _make_inner_env(self, start_level):
        self.current_level = start_level
        env = gym.make(self.env_id,
                       distribution_mode="memory",
                       use_monochrome_assets=False,
                       restrict_themes=True,
                       use_backgrounds=False,
                       num_levels=1,
                       start_level=self.current_level,
                       render_mode=self.render_mode
                       )
        return env

    def _transform_ob(self, ob):
        return np.asarray(ob, dtype=self.obs_dtype) / 255.0

    def render(self, mode="human", **kwargs):
        frame = self._env.render(mode=self.render_mode, **kwargs)
        cv2.imshow("ProcGen Agent", frame)
        cv2.waitKey(1)

    def step(self, action):
        ob, rew, done, info = self._env.step(action)
        return self._transform_ob(ob), rew, done, info

    def reset(self):
        del self._env
        self._env = self._make_inner_env(start_level=self.current_level + 1)
        return self._transform_ob(self._env.reset())

    def seed(self, seed=0):
        # explicitly delete old env to avoid memory leak
        del self._env
        self._env = self._make_inner_env(start_level=seed)


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
            logging.debug("step limit reached")
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
        self._reset_compressor()

    def _reset_compressor(self):
        self.compressed_behavior = b''
        self.compressor = BZ2Compressor(2)
        self.step_count = 0
        self.aggregate = None

    def reset(self, **kwargs):
        return super(BehaviorWrapper, self).reset(**kwargs)

    def _aggregate2compressor(self):
        if self.aggregate is not None:
            data_bytes = np.array(self.aggregate).astype(np.float16).tobytes()
            self.compressed_behavior += self.compressor.compress(data_bytes)
            self.aggregate.fill(0)

    def _record(self, data):
        if self.behavioral_interval < 0:
            # in this case  the actual recording is handled by get_compressed_behavior
            self.aggregate = np.array(data)
            return

        if self.aggregate is None:
            self.aggregate = np.array(data, dtype=np.float32)
            self.aggregate.fill(0)

        if self.behavioral_interval > 0:
            self.aggregate += np.array(data) / self.behavioral_interval

        if self.step_count * self.behavioral_interval < self.behavioral_max_length:
            if self.step_count % self.behavioral_interval == 0:
                self._aggregate2compressor()

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
        if self.behavioral_interval < 0:
            self._aggregate2compressor()
        data = self.compressed_behavior + self.compressor.flush()
        self._reset_compressor()
        return data


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
                if self.unwrapped.MOVEMENTS[inp_act] != "left":
                    rew -= 1

        return ob, rew, done, info
