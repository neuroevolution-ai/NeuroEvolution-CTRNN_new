import abc
from typing import Tuple, Union, TypeVar, Generic, Type, Callable

import numpy as np
import torch.nn as nn
from gym import spaces
from gym.spaces import flatdim

from tools.configurations import IBrainCfg, register

registered_brains = {}


def register_brain(name: str, config_class: Type[IBrainCfg]):
    def _register(type_class: type):
        assert name not in registered_brains, 'type "' + str(
            name) + '" was already registered'
        registered_brains[name] = type_class
        register(name)(
            config_class)  # call to @register-decorator for the config class
        return type_class

    return _register


ConfigClass = TypeVar("ConfigClass", bound=IBrainCfg)


class IBrain(abc.ABC, Generic[ConfigClass]):
    @abc.abstractmethod
    def __init__(self, input_space: spaces.Space, output_space: spaces.Space,
                 individual: np.ndarray, config: ConfigClass):
        self.input_space = input_space
        self.output_space = output_space
        self.config = config

    def step(self, ob: Union[np.ndarray, int, Tuple]):
        """
        Takes an observation from a gym.Environment object and calculates a step of the brain.

        The observation will be transformed according to the input space to a suitable form for the brain, for example
        an observation coming from a Discrete Space will be transformed into a one-hot encoded vector.

        :param ob: Observation from a gym.Environment object
        :return: The output of the step of the Brain, the format of the output corresponds to the output Space of the
                 Brain
        """
        ob = self.parse_brain_input(ob)

        brain_output = self.brainstep(ob)

        return self.parse_brain_output(brain_output)

    @abc.abstractmethod
    def brainstep(self, ob: np.ndarray):
        pass

    @classmethod
    def get_individual_size(cls, config: ConfigClass,
                            input_space: spaces.Space,
                            output_space: spaces.Space) -> int:
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

        usage_dict = cls.get_free_parameter_usage(config, input_space,
                                                  output_space)
        return sum_dict(usage_dict)

    @classmethod
    @abc.abstractmethod
    def get_free_parameter_usage(cls, config: ConfigClass,
                                 input_space: spaces.Space,
                                 output_space: spaces.Space) -> dict:
        """returns a dict for the mapping from free parameters to parts of the brain. For functional purposes
        only the sum of the individual leaves is relevant at the time of writing this docstring.
        Items (and their respective subtrees) are ignored if their key starts with '_'.

        The sum of the values of the leaves of this dict, must be exactly the dimension of the individual, which
        is parsed by the constructor of the respective brain.
        """
        pass

    @staticmethod
    def read_matrix_from_genome(individual: np.ndarray, index: int,
                                matrix_rows: int, matrix_columns: int):
        matrix_size = matrix_columns * matrix_rows

        matrix = np.array(individual[index:index + matrix_size],
                          dtype=np.single)
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
    def set_masks_globally(cls, config: ConfigClass, input_space: spaces.Space,
                           output_space: spaces.Space):
        pass

    def _recursive_parse_brain_input(self, brain_input: Union[np.ndarray, int,
                                                              Tuple],
                                     relevant_space: spaces.Space):
        if isinstance(relevant_space, spaces.Box):
            return brain_input
        elif isinstance(relevant_space, spaces.Discrete):
            # We encode Discrete Spaces as one-hot vectors
            one_hot_vector = np.zeros(relevant_space.n)
            one_hot_vector[brain_input] = 1
            return one_hot_vector
        elif isinstance(relevant_space, spaces.Tuple):
            # Tuple Spaces have a tuple of "nested" Spaces. Environment observations (inputs for the brain) is therefore
            # a tuple where each entry in the tuple is linked to the corresponding Space.
            parsed_brain_input = []
            for i, sub_input in enumerate(brain_input):
                if isinstance(relevant_space[i], spaces.Tuple):
                    raise RuntimeError(
                        """Gym Tuple Spaces inside a gym Tuple space are not supported, maybe choose
                                          another environment""")
                parsed_sub_input = self._recursive_parse_brain_input(
                    sub_input, relevant_space=relevant_space[i])

                # Inputs from the sub spaces are flattened and concatenated here, otherwise some brains could not
                # work with this
                parsed_brain_input = np.concatenate(
                    (parsed_brain_input, parsed_sub_input.flatten()))
            return np.array(parsed_brain_input)

        else:
            raise RuntimeError(
                "The gym.Space '{}' is currently not supported to be parsed.".
                format(type(relevant_space)))

    def parse_brain_input(self, brain_input: Union[np.ndarray, int, Tuple]):
        """
        Pre-processes the input for the brain according to the gym.Space it comes from.

        If the input space is a:
            - spaces.Box then nothing is done to brain_input
            - spaces.Discrete then brain_input is encoded as a one-hot vector
            - spaces.Tuple then the brain_input is a tuple of inputs where each entry corresponds to a sub-space inside
                the Tuple Space. These entries get recursively pre-processed and concatenated into a one-dimensional
                NumPy array.

        Other Spaces or Tuple Spaces inside Tuple Spaces are currently not supported.

        :param brain_input: An observation from an environment
        :return: Pre-processed observation, according to self.input_space
        """
        # We are coming from the input space therefore set relevant space accordingly
        return self._recursive_parse_brain_input(
            brain_input, relevant_space=self.input_space)

    def _recursive_parse_brain_output(self, brain_output: np.ndarray,
                                      relevant_space: spaces.Space):
        if isinstance(relevant_space, spaces.Box):
            return brain_output.reshape(relevant_space.shape)
        elif isinstance(relevant_space, spaces.Discrete):
            return np.argmax(brain_output)
        elif isinstance(relevant_space, spaces.Tuple):
            # Transform the brain output into a tuple which match the corresponding "nested" Spaces
            parsed_brain_output = []
            index = 0
            for sub_space in relevant_space:
                current_range = flatdim(sub_space)

                parsed_brain_output.append(
                    self._recursive_parse_brain_output(
                        brain_output[index:index + current_range],
                        relevant_space=sub_space))
                index += current_range
            return tuple(parsed_brain_output)
        else:
            raise RuntimeError(
                "The gym.Space '{}' is currently not supported to be parsed.".
                format(type(relevant_space)))

    def parse_brain_output(self, brain_output: np.ndarray):
        """
        Post-processes the output of the brain so that it can be applied in the output space (i.e. the action space of
        the environment).

        If the output space is a:
            - spaces.Box then brain_output is reshaped to the shape of self.output_space
            - spaces.Discrete then the argmax is taken from brain_output
            - spaces.Tuple then brain_output is reshaped into a tuple where each entry of the tuple is recursively
                created to match its sub-space.

        Other Spaces are currently not supported.

        :param brain_output: The output that is coming from the brain
        :return: Post-processed brain output, according to self.output_space
        """
        # We are going to the output space
        return self._recursive_parse_brain_output(
            brain_output, relevant_space=self.output_space)

    @classmethod
    def generate_and_set_class_state(cls, config: ConfigClass,
                                     input_space: spaces.Space,
                                     output_space: spaces.Space):
        pass

    def get_neuron_states(self):
        raise NotImplementedError(
            "Support for brain visualizer is not implemented for this brain.")

    def get_internal_weight_matrix(self):
        raise NotImplementedError(
            "Support for brain visualizer is not implemented for this brain.")

    def get_input_weight_matrix(self):
        raise NotImplementedError(
            "Support for brain visualizer is not implemented for this brain.")

    def get_output_weight_matrix(self):
        raise NotImplementedError(
            "Support for brain visualizer is not implemented for this brain.")

    @classmethod
    def get_activation_function_numpy(
            cls, activation: str) -> Callable[[np.ndarray], np.ndarray]:

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
    def get_activation_function_pytorch(
            cls, activation: str) -> Callable[[np.ndarray], np.ndarray]:

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
