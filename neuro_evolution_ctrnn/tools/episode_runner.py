from bz2 import BZ2Compressor
import gym
import logging
import numpy as np
import time
from typing import Optional

from tools.helper import set_random_seeds, output_to_action
from tools.configurations import EpisodeRunnerCfg, IBrainCfg
from tools.env_handler import EnvHandler
import cv2


class EpisodeRunner:
    _env: Optional[gym.Env] = None

    def __init__(self, config: EpisodeRunnerCfg, brain_config: IBrainCfg, brain_class, input_space, output_space,
                 env_id):
        self.config = config
        self.brain_config = brain_config
        self.brain_class = brain_class
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_id
        self.env_handler = EnvHandler(self.config)

    def _get_env(self, record, record_force, render):
        if self.config.reuse_env:
            if EpisodeRunner._env is None:
                EpisodeRunner._env = env = self.env_handler.make_env(self.env_id, render=render, record=record,
                                                                     record_force=record_force)
            else:
                env = EpisodeRunner._env
                # split is needed for the procgen environments
                assert self.env_id.split(":")[-1] == EpisodeRunner._env.spec.id
        else:
            env = self.env_handler.make_env(self.env_id, render=render, record=record, record_force=record_force)

        return env

    def _render(self, env, ob, render_raw_ob):
        if render_raw_ob:
            # only looks good if output is RGB Data
            cv2.imshow("Agent's Observation", ob.astype(np.float32))
            cv2.waitKey(1)
        else:
            env.render()

    def eval_fitness(self, individual, seed, render: bool = False, record: str = None, record_force: bool = False,
                     brain_vis_handler=None, neuron_vis=False, slow_down=0, rounds=None, neuron_vis_width=None,
                     neuron_vis_height=None, render_raw_ob=False):
        env = self._get_env(record, record_force, render)
        set_random_seeds(seed, env)
        fitness_total = 0
        steps_total = 0
        number_of_rounds = self.config.number_fitness_runs if rounds is None else rounds
        for i in range(number_of_rounds):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual, self.brain_config)
            ob = env.reset()
            done = False
            t = 0

            if render:
                self._render(env, ob, render_raw_ob)

            if neuron_vis:
                brain_vis = brain_vis_handler.launch_new_visualization(brain=brain, brain_config=self.brain_config,
                                                                       env_id=self.env_id, initial_observation=ob,
                                                                       width=neuron_vis_width, height=neuron_vis_height,
                                                                       color_clipping_range=(255, 2.5, 2.5),
                                                                       slow_down=slow_down)
            else:
                brain_vis = None

            while not done:
                brain_output = brain.step(ob)
                action = output_to_action(brain_output, self.output_space)
                ob, rew, done, info = env.step(action)
                t += 1
                fitness_current += rew

                if brain_vis:
                    brain_vis.process_update(in_values=ob, out_values=brain_output)
                if slow_down:
                    time.sleep(slow_down / 1000.0)
                if render:
                    self._render(env, ob, render_raw_ob)
            if render:
                logging.info("steps: " + str(t) + " \tfitness: " + str(fitness_current))

            fitness_total += fitness_current
            steps_total += t

        compressed_behavior = None
        if hasattr(env, 'get_compressed_behavior'):
            # 'get_compressed_behavior' exists if any wrapper is a BehaviorWrapper
            assert callable(env.get_compressed_behavior)
            compressed_behavior = env.get_compressed_behavior()

        return fitness_total / number_of_rounds, compressed_behavior, steps_total
