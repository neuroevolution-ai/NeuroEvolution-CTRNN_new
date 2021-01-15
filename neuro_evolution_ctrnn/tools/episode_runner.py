import logging
import time
from bz2 import BZ2Compressor
from typing import Optional, Tuple, Union

import gym
import numpy as np

from brains.CNN_CTRNN import CnnCtrnn
from brains.continuous_time_rnn import ContinuousTimeRNN
from tools.configurations import EpisodeRunnerCfg, IBrainCfg
from tools.env_handler import EnvHandler
from tools.helper import set_random_seeds


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

    def eval_fitness(self, individual, seed, render: bool = False, record: str = None, record_force: bool = False,
                     brain_vis_handler=None, neuron_vis=False, slow_down=0, rounds=None, neuron_vis_width=None,
                     neuron_vis_height=None):
        env = self._get_env(record, record_force, render)
        set_random_seeds(seed, env)
        fitness_total = 0
        steps_total = 0
        number_of_rounds = self.config.number_fitness_runs if rounds is None else rounds
        brain_state_history = []
        for i in range(number_of_rounds):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual, self.brain_config)
            ob = env.reset()
            transformed_ob = self.transform(ob, coming_from_space=self.input_space, is_brain_input=True)
            # TODO normalize
            done = False
            t = 0

            if render:
                env.render()

            if neuron_vis:
                # TODO BrainVis use transformed_ob, dont forget update step in training loop
                brain_vis = brain_vis_handler.launch_new_visualization(brain=brain, brain_config=self.brain_config,
                                                                       env_id=self.env_id, initial_observation=ob,
                                                                       width=neuron_vis_width, height=neuron_vis_height,
                                                                       color_clipping_range=(255, 2.5, 2.5),
                                                                       slow_down=slow_down)
            else:
                brain_vis = None

            while not done:
                brain_output = brain.step(transformed_ob)
                action = self.transform(brain_output, coming_from_space=self.output_space, is_brain_input=False)
                ob, rew, done, info = env.step(action)
                transformed_ob = self.transform(ob, coming_from_space=self.input_space, is_brain_input=True)
                t += 1
                fitness_current += rew

                if brain_vis:
                    brain_vis.process_update(in_values=ob, out_values=brain_output)
                if slow_down:
                    time.sleep(slow_down / 1000.0)
                if render:
                    env.render()
                if self.config.novelty:
                    if self.config.novelty.behavior_source == 'brain':
                        if isinstance(brain, ContinuousTimeRNN):
                            brain_state_history.append(np.tanh(brain.y))
                        elif isinstance(brain, CnnCtrnn):
                            brain_state_history.append(np.tanh(brain.ctrnn.y))
                        else:
                            logging.error('behavior_source == "brain" not yet supported for this kind of brain')

            if render:
                logging.info("steps: " + str(t) + " \tfitness: " + str(fitness_current))

            fitness_total += fitness_current
            steps_total += t
            # print(info['level_seed'])

        compressed_behavior = None
        if hasattr(env, 'get_compressed_behavior'):
            # 'get_compressed_behavior' exists if any wrapper is a BehaviorWrapper
            if callable(env.get_compressed_behavior):
                compressed_behavior = env.get_compressed_behavior()

        if self.config.novelty:
            if self.config.novelty.behavior_source == 'brain':
                # todo: remove code duplication. This code is also in BehaviorWrapper
                compressor = BZ2Compressor(2)
                compressed_behavior = b''
                if self.config.novelty.behavioral_max_length < 0:
                    compressor.compress(brain_state_history[-1].astype(np.float16).tobytes())
                    compressed_behavior += compressor.flush()
                else:
                    for i in range(self.config.novelty.behavioral_max_length):
                        aggregate = np.zeros(len(brain_state_history[0]), dtype=np.float32)
                        for j in range(self.config.novelty.behavioral_interval):
                            if len(brain_state_history) > j + i * self.config.novelty.behavioral_interval:
                                state = brain_state_history[j + i * self.config.novelty.behavioral_interval]
                                aggregate += state / self.config.novelty.behavioral_interval
                            else:
                                break
                        compressed_behavior += compressor.compress(aggregate.astype(np.float16).tobytes())
                    compressed_behavior += compressor.flush()

        return fitness_total / number_of_rounds, compressed_behavior, steps_total

    def transform(
            self,
            to_transform: Union[np.ndarray, int, Tuple],
            coming_from_space: gym.Space,
            is_brain_input: bool) -> Union[np.ndarray, int, Tuple]:
        """
        Transforms 'to_transform' to a new 'format'. If is_brain_input is True this is a format which is accepted by a
        Brain, i.e. a one-dimensional np.ndarray. If is_brain_input is False the format corresponds to the provided Space,
        i.e. the action space of the environment.

        An input for the brain equals the output of an environment and vice versa.

        :param to_transform: The to be transformed variable
        :param coming_from_space: The Space from which to_transform comes from. If is_brain_input is True this is the
            observation space of the environment, if it is False this is the action space
        :param is_brain_input: True if to_transform is the input for a Brain, False if not
        :return: The transformed variable
        """
        # TODO BrainVisualizer transform, remove code for normalizing from brains remove old function for transform from
        #  helper.py and its usages
        # TODO better name for function maybe transform_brain_input_output
        if isinstance(coming_from_space, gym.spaces.Box):
            return to_transform
        elif isinstance(coming_from_space, gym.spaces.Discrete):
            # We encode Discrete Spaces as one-hot vectors
            if is_brain_input:
                one_hot_vector = np.zeros(coming_from_space.n)
                one_hot_vector[to_transform] = 1
                return one_hot_vector
            else:
                return np.argmax(to_transform)
        elif isinstance(coming_from_space, gym.spaces.Tuple):
            # Tuple Spaces have a tuple of "nested" Spaces. Environment observations (inputs for the brain) is therefore
            # a tuple where each entry in the tuple is linked to the corresponding Space.
            if is_brain_input:
                # Transform the brain input (a tuple) into a one-dimensional np.ndarray
                brain_input = []
                for i, sub_input in enumerate(to_transform):
                    brain_input = np.concatenate(
                        (brain_input, transform(sub_input, coming_from_space=coming_from_space[i], is_brain_input=True))
                    )
                return brain_input
            else:
                # Transform the brain output into a tuple which match the corresponding "nested" Spaces
                brain_output = []
                index = 0
                for sub_space in coming_from_space:
                    # Because Discrete spaces are encoded as one-hot vectors we cannot take their shape to determine
                    # current range, therefore the n attribute is used
                    if isinstance(sub_space, gym.spaces.Box):
                        current_range = np.prod(sub_space.shape)
                    elif isinstance(sub_space, gym.spaces.Discrete):
                        current_range = sub_space.n
                    else:
                        raise RuntimeError(
                            "'{}' is not supported to be transformed inside a Tuple Space, use Box or Discrete."
                            "".format(sub_space.__name__)
                        )

                    brain_output.append(
                        transform(
                            to_transform[index:index + current_range], coming_from_space=sub_space, is_brain_input=False
                        )
                    )
                    index += current_range

                assert index == to_transform.size

                return tuple(brain_output)
        else:
            raise RuntimeError(
                "The gym.Space '{}' is currently not supported to be transformed.".format(coming_from_space.__name__)
            )
