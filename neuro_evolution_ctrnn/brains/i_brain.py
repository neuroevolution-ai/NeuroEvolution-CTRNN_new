import abc
from tools.configurations import IBrainCfg, register
import numpy as np
from gym.spaces import Space, Discrete, Box, tuple
from typing import TypeVar, Generic, get_args, Type

registered_brains = {}


def register_brain(name: str, config_class: Type[IBrainCfg]):
    def _register(type_class: type):
        assert name not in registered_brains, 'type "' + str(name) + '" was already registered'
        registered_brains[name] = type_class
        register(name)(config_class)
        return type_class

    return _register


ConfigClass = TypeVar('ConfigClass', bound=IBrainCfg)


class IBrain(abc.ABC, Generic[ConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        # todo: remove these default assignments from derived classes
        self.input_space = input_space
        self.output_space = output_space
        self.config = config

    def step(self, ob: np.ndarray):
        return self.brainstep(self.prepare_input(ob))

    def prepare_input(self, ob: np.ndarray):
        if self.config.normalize_input:
            return self._normalize_input(ob, self.input_space, self.config.normalize_input_target)
        else:
            return ob

    @abc.abstractmethod
    def brainstep(self, ob: np.ndarray):
        pass

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space) -> int:
        """uses context information to calculate the required number of free parameter needed to construct
                an individual of this class"""

        def sum_dict(node):
            sum_ = 0
            for key, item in node.items():
                if str(key).startswith('_'):
                    continue
                if isinstance(item, dict):
                    sum_ += sum_dict(item)
                else:
                    sum_ += item
            return sum_

        usage_dict = cls.get_free_parameter_usage(config, input_space, output_space)
        return sum_dict(usage_dict)

    @classmethod
    @abc.abstractmethod
    def get_free_parameter_usage(cls, config: ConfigClass, input_space: Space, output_space: Space) -> dict:
        """returns a dict for the mapping from free parameters to parts of the brain. For functional purposes
        only the sum of the individual leaves is relevant at the time of writing this docstring.
        Items (and their respective subtrees) are ignored if their key starts with '_'.

        The sum of the values of the leaves of this dict, must be exactly the dimension of the individual, which
        is parsed by the constructor of the respective brain.
        """
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
        return (x - low) / (high - low)

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

            # ob[mask] is now between -target and +target
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
            sum_ = 0
            for x in space:
                sum_ += IBrain._size_from_space(x)
            return sum_
        else:
            raise NotImplementedError("not implemented input/output space: " + str(type(space)))

    @classmethod
    def generate_and_set_class_state(cls, config: ConfigClass, input_space: Space, output_space: Space):
        pass

    def discrete_to_vector(self, ob):
        # todo: use this function in all derived classes
        #  and maybe find a even cleaner way to do this without violating DRY
        ob_new = np.zeros(self.input_space.n)
        ob_new[ob] = 1
        return ob_new

    def get_neuron_states(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    def get_internal_weight_matrix(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    def get_input_weight_matrix(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    def get_output_weight_matrix(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")
