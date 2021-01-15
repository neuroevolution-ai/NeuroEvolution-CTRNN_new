import abc
from typing import Generic, Tuple, Union, TypeVar

import numpy as np
from gym import spaces

from tools.configurations import IBrainCfg

ConfigClass = TypeVar("ConfigClass", bound=IBrainCfg)


class IBrain(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: spaces.Space, output_space: spaces.Space, individual: np.ndarray,
                 config: ConfigClass):
        self.input_space = input_space
        self.output_space = output_space
        self.config = config

    def step(self, ob: np.ndarray):
        """
        Takes an observation from a gym.Environment object and calculates a step of the brain.

        If config.normalize_input is set the observations get normalized before the calculations inside the brain
        are done.

        :param ob: Observation from a gym.Environment object
        :return: The output of the step of the Brain, the format of the output corresponds to the output_space variable
            of the Brain
        """
        if self.config.normalize_input:
            ob = self._scale_observation(ob)

        ob = self.parse_brain_input_output(ob, is_brain_input=True)

        brain_output = self.calculate_brain_output(ob)

        return self.parse_brain_input_output(brain_output, is_brain_input=False)

    @abc.abstractmethod
    def calculate_brain_output(self, ob: np.ndarray):
        pass

    @classmethod
    @abc.abstractmethod
    def get_individual_size(cls, config: ConfigClass, input_space: spaces.Space, output_space: spaces.Space):
        pass

    @classmethod
    def get_class_state(cls):
        return {}

    @classmethod
    def set_class_state(cls, *nargs, **kwargs):
        pass

    @classmethod
    def set_masks_globally(cls, config: ConfigClass, input_space: spaces.Space, output_space: spaces.Space):
        pass

    @staticmethod
    def _normalize(x, low, high):
        """Scales a value x from the interval [low,high] to the interval [0,1]."""
        return ((x - low) / (high - low))

    def _scale_observation(self, ob):
        if isinstance(self.input_space, spaces.Box):
            # Note: some Spaces from some envs do not define input_space.bounded_below properly so we treat
            # very high bounds as unbounded, too
            mask = (self.input_space.bounded_below
                    & self.input_space.bounded_above
                    & (self.input_space.bounded_below > -1e5)
                    & (self.input_space.bounded_above < 1e5))

            # scaled is now between 0 and 1
            scaled = self._normalize(ob[mask], self.input_space.low[mask], self.input_space.high[mask])

            # ob[mask] is now between -target and +target
            ob[mask] = (scaled - 0.5) * (2 * self.config.normalize_input_target)

            return ob
        else:
            raise NotImplementedError("_scale_observation is only implemented for the input Spaces of type Box.")

    @staticmethod
    def _size_from_space(space: spaces.Space) -> int:
        if isinstance(space, spaces.Discrete):
            return space.n  # type: ignore
        elif isinstance(space, spaces.Box):
            return np.prod(space.shape)  # type: ignore
        elif isinstance(space, spaces.Tuple):
            size = 0
            for x in space:
                size += IBrain._size_from_space(x)
            return size
        else:
            raise NotImplementedError("Calculating the size is not implemented for Spaces of type '{}'"
                                      "".format(type(space)))

    def parse_brain_input_output(self,
                                 brain_input_output: Union[np.ndarray, int, Tuple],
                                 is_brain_input: bool) -> Union[np.ndarray, int, Tuple]:
        """
        Parses 'brain_input_output' to a new 'format'. If is_brain_input is True this is a np.ndarray. If is_brain_input
        is False the format corresponds to the output Space, i.e. the action space of the environment.

        An input for the brain equals the output of an environment and vice versa.

        :param brain_input_output: The to be parsed variable
        :param is_brain_input: True if to_transform is the input for a Brain, False if not
        :return: The parsed variable
        """
        coming_from_space = self.input_space if is_brain_input else self.output_space

        if isinstance(coming_from_space, spaces.Box):
            return brain_input_output
        elif isinstance(coming_from_space, spaces.Discrete):
            # We encode Discrete Spaces as one-hot vectors
            if is_brain_input:
                one_hot_vector = np.zeros(coming_from_space.n)
                one_hot_vector[brain_input_output] = 1
                return one_hot_vector
            else:
                return np.argmax(brain_input_output)
        elif isinstance(coming_from_space, spaces.Tuple):
            # Tuple Spaces have a tuple of "nested" Spaces. Environment observations (inputs for the brain) is therefore
            # a tuple where each entry in the tuple is linked to the corresponding Space.
            if is_brain_input:
                # Transform the brain input (a tuple) into a one-dimensional np.ndarray
                brain_input = []
                for i, sub_input in enumerate(brain_input_output):
                    brain_input = np.concatenate(
                        (brain_input, self.parse_brain_input_output(sub_input, is_brain_input=True))
                    )
                return brain_input
            else:
                # Transform the brain output into a tuple which match the corresponding "nested" Spaces
                brain_output = []
                index = 0
                for sub_space in coming_from_space:
                    current_range = self._size_from_space(sub_space)

                    brain_output.append(
                        self.parse_brain_input_output(
                            brain_input_output[index:index + current_range], is_brain_input=False
                        )
                    )
                    index += current_range

                assert index == brain_input_output.size

                return tuple(brain_output)
        else:
            raise RuntimeError(
                "The gym.Space '{}' is currently not supported to be parsed.".format(type(coming_from_space))
            )
