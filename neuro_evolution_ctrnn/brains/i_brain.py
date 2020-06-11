import abc
from tools.configurations import IBrainCfg
import numpy as np
from gym.spaces import Space, Discrete, Box
from typing import TypeVar, Generic

ConfigClass = TypeVar('ConfigClass', bound=IBrainCfg)


class IBrain(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        pass

    @abc.abstractmethod
    def step(self, ob: np.ndarray):
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
        """scales a value x from interval [low,high] to interval [0,1]"""
        return ((x - low) / (high - low))

    @staticmethod
    def _scale_observation(ob, input_space:Space, target:float):

        if isinstance(input_space, Box):
            # note: some spaces from some envs don't define input_space.bounded_below properly so we treat
            # very high bounds as unbounded, too
            mask = input_space.bounded_below & input_space.bounded_above \
                   & (input_space.bounded_below > -1e5) \
                   & (input_space.bounded_above < 1e5)

            # scaled is now between 0 and 1
            scaled = IBrain._normalize(ob[mask], input_space.low[mask],
                                     input_space.high[mask])

            # ob[mask] is now betwen -target and +target
            ob[mask] = (scaled - 0.5) * (2 * target)
        else:
            raise NotImplementedError("normalize_input is only defined for input-type Box")
        return ob

    @staticmethod
    def _size_from_space(space: Space) -> int:
        if isinstance(space, Discrete):
            return space.n  # type: ignore
        elif isinstance(space, Box):
            return np.prod(space.shape)  # type: ignore
        else:
            raise NotImplementedError("not implemented input/output space: " + str(type(space)))

    @classmethod
    def set_masks_globally(cls, config: ConfigClass, input_space: Space, output_space: Space):
        pass
