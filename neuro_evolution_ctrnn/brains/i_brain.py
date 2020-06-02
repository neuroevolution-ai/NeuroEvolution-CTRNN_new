import abc
from tools.configurations import IBrainCfg
import numpy as np
from gym.spaces import Space, Discrete, Box
from typing import Type, TypeVar


ConfigClass = TypeVar('ConfigClass', bound=IBrainCfg)

class IBrain(abc.ABC):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: IBrainCfg):
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
