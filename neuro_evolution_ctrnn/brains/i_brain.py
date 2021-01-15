import abc
from typing import TypeVar, Generic

import numpy as np
from gym.spaces import Space, Discrete, Box, tuple

from tools.configurations import IBrainCfg

ConfigClass = TypeVar("ConfigClass", bound=IBrainCfg)


class IBrain(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        self.input_space = input_space
        self.output_space = output_space
        self.config = config

    def step(self, ob: np.ndarray):
        if self.config.normalize_input:
            ob = self._scale_observation(ob)

        return self.calculate_brain_output(ob)

    @abc.abstractmethod
    def calculate_brain_output(self, ob: np.ndarray):
        pass

    @classmethod
    @abc.abstractmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        pass

    @classmethod
    def get_class_state(cls):
        return {}

    @classmethod
    def set_class_state(cls, *nargs, **kwargs):
        pass

    @staticmethod
    def _normalize(x, low, high):
        """Scales a value x from the interval [low,high] to the interval [0,1]."""
        return ((x - low) / (high - low))

    def _scale_observation(self, ob):
        if isinstance(self.input_space, Box):
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
    def _size_from_space(space: Space) -> int:
        if isinstance(space, Discrete):
            return space.n  # type: ignore
        elif isinstance(space, Box):
            return np.prod(space.shape)  # type: ignore
        elif isinstance(space, tuple.Tuple):
            sum = 0
            for x in space:
                sum += IBrain._size_from_space(x)
            return sum
        else:
            raise NotImplementedError("not implemented input/output space: " + str(type(space)))

    @classmethod
    def set_masks_globally(cls, config: ConfigClass, input_space: Space, output_space: Space):
        pass
