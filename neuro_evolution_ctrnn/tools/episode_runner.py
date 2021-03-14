import logging
import time
from typing import Optional

import cv2
import gym
import numpy as np

from tools.configurations import EpisodeRunnerCfg, IBrainCfg
from tools.env_handler import EnvHandler
from tools.helper import set_random_seeds
from bz2 import decompress


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

    def eval_fitness(self, individual, seed, novelty_data, size_mutation_genes: int, render: bool = False,
                     record: str = None, record_force: bool = False,
                     brain_vis_handler=None, neuron_vis=False, slow_down=0, rounds=None, neuron_vis_width=None,
                     neuron_vis_height=None, render_raw_ob=False):
        env = self._get_env(record, record_force, render)
        set_random_seeds(seed, env)
        fitness_total = 0
        steps_total = 0
        number_of_rounds = self.config.number_fitness_runs if rounds is None else rounds
        for i in range(number_of_rounds):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space,
                                     individual[:-size_mutation_genes] if size_mutation_genes else individual,
                                     self.brain_config)
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
                action = brain.step(ob)
                ob, rew, done, info = env.step(action)
                t += 1
                fitness_current += rew
                if brain_vis:
                    brain_vis.process_update(in_values=ob, out_values=action)
                if slow_down:
                    time.sleep(slow_down / 1000.0)
                if render:
                    self._render(env, ob, render_raw_ob)
            if render:
                logging.info("steps: " + str(t) + " \tfitness: " + str(fitness_current))

            fitness_total += fitness_current
            steps_total += t

        # TODO does this gets used anywhere?
        # compressed_behavior = None
        # if hasattr(env, "get_compressed_behavior"):
        #     # 'get_compressed_behavior' exists if any wrapper is a BehaviorWrapper
        #     assert callable(env.get_compressed_behavior)
        #     compressed_behavior = env.get_compressed_behavior()

        individual.novelty = self.calc_novelty(*novelty_data) if novelty_data else 0

        individual.fitness_orig = fitness_total / number_of_rounds
        individual.steps = steps_total
        return individual, steps_total

    @staticmethod
    def calc_novelty(res, results_recorded, get_distance, k):
        """calculate the average distance to the k nearest neighbors"""
        if k < 1:
            k = len(results_recorded)
        behavior_compressed = res[1]
        behavior_data = decompress(behavior_compressed)
        behavior = np.frombuffer(behavior_data, dtype=np.float16)
        dist_list = []
        k = min(k, len(results_recorded))
        for rec_res in results_recorded:
            recorded_behavior_compressed = rec_res[1]
            recorded_behavior_data = decompress(recorded_behavior_compressed)
            recorded_behavior = np.frombuffer(recorded_behavior_data, dtype=np.float16)
            dist = get_distance(a=behavior, b=recorded_behavior, a_len=len(behavior_compressed),
                                b_len=len(recorded_behavior_compressed))
            dist_list.append(dist)

        dist_sum = 0
        for nearest_neighbor in sorted(dist_list, reverse=False)[0:k]:
            dist_sum += nearest_neighbor
        return dist_sum / k
