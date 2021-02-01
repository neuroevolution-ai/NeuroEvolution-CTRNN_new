import abc
from tools.configurations import IBrainCfg
from tools.helper import walk_dict
import numpy as np
from gym.spaces import Space, Discrete, Box, tuple
from typing import TypeVar, Generic

ConfigClass = TypeVar('ConfigClass', bound=IBrainCfg)


class IBrain(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        # todo: remove these default assignments from derived classes
        self.input_space = input_space
        self.output_space = output_space
        self.config = config

    @abc.abstractmethod
    def step(self, ob: np.ndarray):
        pass

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space) -> int:
        """uses context information to calculate the required number of free parameter needed to construct
                an individual of this class"""

        def add(key, item, depth, is_leaf):
            nonlocal sum_

            if str(key).startswith('_'):
                return
            if is_leaf:
                sum_ += item

        slice_dict = cls.get_individual_slices(config, input_space, output_space)
        sum_ = 0
        walk_dict(slice_dict, add)
        return sum_

    @classmethod
    @abc.abstractmethod
    def get_individual_slices(cls, config: ConfigClass, input_space: Space, output_space: Space) -> dict:
        """returns a dict for the mapping from free parameters to parts of the brain. The keys
         are ignored for functional purposes, and only serve as hint for the user. Items are ignored for
         functional purposes if their key starts with '_'. """
        pass

    @classmethod
    def get_class_state(cls):
        return {}

    @classmethod
    def set_class_state(cls, *nargs, **kwargs):
        pass

    @staticmethod
    def _normalize_input(ob, input_space, normalize_input_target):
        # todo: remove this
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
        """scales a value x from interval [low,high] to interval [0,1]"""
        return ((x - low) / (high - low))

    @staticmethod
    def _scale_observation(ob, input_space: Space, target: float):
        # todo: remove this
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

    def discrete_to_vector(self, ob):
        # todo: use this function in all derived classes
        #  and maybe find a even cleaner way to do this without violating DRY
        ob_new = np.zeros(self.input_space.n)
        ob_new[ob] = 1
        return ob_new
