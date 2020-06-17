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
    def _normalize_input(ob, input_space, normalize_input_target):
        for idx, item in enumerate(ob):
            if isinstance(input_space, Box):
                if input_space.bounded_below[idx] and input_space.bounded_above[idx]:
                    ob[idx] = IBrain._normalize(ob[idx], input_space.low[idx],
                                                input_space.high[idx]) * normalize_input_target
            else:
                raise NotImplementedError("normalize_input is only defined for input-type Box")

        return ob

    @staticmethod
    def _normalize(x, low, high):
        if high > 1e5 or low < -1e5:
            # treat high spaces as unbounded
            # note: some spaces some envs don't define input_space.bounded_below properly
            return x
        return (((x - low) / (high - low)) * 2) - 1

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
