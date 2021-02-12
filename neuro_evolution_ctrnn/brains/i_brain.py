import abc
from typing import Tuple, Union, TypeVar, Generic, Type, Callable

import numpy as np
import torch.nn as nn
from gym import spaces

from tools.configurations import IBrainCfg, register

registered_brains = {}


def register_brain(name: str, config_class: Type[IBrainCfg]):
    def _register(type_class: type):
        assert name not in registered_brains, 'type "' + str(name) + '" was already registered'
        registered_brains[name] = type_class
        register(name)(config_class)  # call to @register-decorator for the config class
        return type_class

    return _register


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

        The observation will be transformed according to the input space to a suitable form for the brain, for example
        an observation coming from a Discrete Space will be transformed into a one-hot encoded vector.

        :param ob: Observation from a gym.Environment object
        :return: The output of the step of the Brain, the format of the output corresponds to the output Space of the
                 Brain
        """
        ob = self._flatten(ob)

        brain_output = self.brainstep(ob)

        return self._unflatten(brain_output)

    def _flatten(self, ob: Union[np.ndarray, int, Tuple]):
        # Simply pass through input of a Box Space because the util function from gym completely flattens ob
        # For RGB input we would like to retain these input shapes
        if isinstance(self.input_space, spaces.Box):
            return ob
        else:
            return spaces.flatten(self.input_space, ob)

    def _unflatten(self, brain_output: np.ndarray):
        # If the output Space is of type Discrete take the argmax of the brain_output. The unflatten util function
        # from the gym would take the first non-zero element of brain_output which is not desired for us
        if isinstance(self.output_space, spaces.Discrete):
            return np.argmax(brain_output)
        else:
            return spaces.unflatten(self.output_space, brain_output)

    @abc.abstractmethod
    def brainstep(self, ob: np.ndarray):
        pass

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: spaces.Space, output_space: spaces.Space) -> int:
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
    def get_free_parameter_usage(cls, config: ConfigClass, input_space: spaces.Space,
                                 output_space: spaces.Space) -> dict:
        """returns a dict for the mapping from free parameters to parts of the brain. For functional purposes
        only the sum of the individual leaves is relevant at the time of writing this docstring.
        Items (and their respective subtrees) are ignored if their key starts with '_'.

        The sum of the values of the leaves of this dict, must be exactly the dimension of the individual, which
        is parsed by the constructor of the respective brain.
        """
        pass

    @staticmethod
    def read_matrix_from_genome(individual: np.ndarray, index: int, matrix_rows: int, matrix_columns: int):
        matrix_size = matrix_columns * matrix_rows

        matrix = np.array(individual[index:index + matrix_size], dtype=np.single)
        matrix = matrix.reshape(matrix_rows, matrix_columns)

        index += matrix_size

        return matrix, index

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

    @classmethod
    def generate_and_set_class_state(cls, config: ConfigClass, input_space: spaces.Space, output_space: spaces.Space):
        pass

    def get_neuron_states(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    def get_internal_weight_matrix(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    def get_input_weight_matrix(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    def get_output_weight_matrix(self):
        raise NotImplementedError("Support for brain visualizer is not implemented for this brain.")

    @classmethod
    def get_activation_function_numpy(cls, activation: str) -> Callable[[np.ndarray], np.ndarray]:

        if activation == "relu":
            return cls.relu
        elif activation == "linear":
            return cls.linear
        elif activation == "tanh":
            return cls.tanh
        else:
            raise RuntimeError(
                "The chosen non linearity '{}' is not implemented, choose either 'relu' or 'tanh'"
                "".format(activation))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @classmethod
    def get_activation_function_pytorch(cls, activation: str) -> Callable[[np.ndarray], np.ndarray]:

        if activation == "relu":
            return nn.ReLU()  # type:ignore
        elif activation == "linear":
            return nn.Identity()  # type:ignore
        elif activation == "tanh":
            return nn.Tanh()  # type:ignore
        else:
            raise RuntimeError(
                "The chosen non linearity '{}' is not implemented, choose either 'relu' or 'tanh'"
                "".format(activation))
