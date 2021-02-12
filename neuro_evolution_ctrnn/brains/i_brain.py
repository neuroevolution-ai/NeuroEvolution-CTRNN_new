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
        ob = self.parse_brain_input_output(ob, is_brain_input=True)

        brain_output = self.brainstep(ob)

        return self.parse_brain_input_output(brain_output, is_brain_input=False)

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

    def parse_brain_input_output(self,
                                 brain_input_output: Union[np.ndarray, int, Tuple],
                                 is_brain_input: bool,
                                 relevant_space: spaces.Space = None) -> Union[np.ndarray, int, Tuple, list]:
        """
        Parses 'brain_input_output' to a new 'format'. If is_brain_input is True this is a np.ndarray. If is_brain_input
        is False the format corresponds to the output Space, i.e. the action space of the environment.

        An input for the brain equals the output of an environment and vice versa.

        Discrete Space as input space will be encoded as a one hot vector, Discrete Space as output takes the argmax.
        Box Space as input will not be transformed, Box Space as output will be reshaped to match the environment.
        Tuple Space as input unpacks each subspace, transforms it with this function (recursive) and then flattens every
        output of the subspaces. For Tuple Space as output a tuple will be created from the brain output where the
        entries are of the correct shape for the environment.

        Note that Tuple Spaces inside Tuple Spaces are not supported.

        :param brain_input_output: The to be parsed variable
        :param is_brain_input: Should be True if brain_input_output is the input for a Brain, False if not
        :param relevant_space: relevant_space is normally chosen depending on is_brain_input but it can be provided
        directly with this parameter. This is used for Tuple spaces which are nested
        :return: The parsed variable
        """
        # This is the Space we need to calculate the correct output form
        if relevant_space is None:
            relevant_space = self.input_space if is_brain_input else self.output_space

        if isinstance(relevant_space, spaces.Box):
            if is_brain_input:
                return brain_input_output
            else:
                return brain_input_output.reshape(relevant_space.shape)
        elif isinstance(relevant_space, spaces.Discrete):
            # We encode Discrete Spaces as one-hot vectors
            if is_brain_input:
                one_hot_vector = np.zeros(relevant_space.n)
                one_hot_vector[brain_input_output] = 1
                return one_hot_vector
            else:
                return np.argmax(brain_input_output)
        elif isinstance(relevant_space, spaces.Tuple):
            # Tuple Spaces have a tuple of "nested" Spaces. Environment observations (inputs for the brain) is therefore
            # a tuple where each entry in the tuple is linked to the corresponding Space.
            if is_brain_input:
                # Transform the brain input (a tuple) into a list which is structured to resemble the "nested" Spaces.
                brain_input = []
                for i, sub_input in enumerate(brain_input_output):
                    if isinstance(relevant_space[i], spaces.Tuple):
                        raise RuntimeError("""Gym Tuple Spaces inside a gym Tuple space are not supported, maybe choose
                                              another environment""")
                    parsed_sub_input = self.parse_brain_input_output(sub_input,
                                                                     is_brain_input=True,
                                                                     relevant_space=relevant_space[i])

                    # Inputs from the sub spaces are flattened and concatenated here, otherwise some brains could not
                    # work with this
                    brain_input = np.concatenate((brain_input, parsed_sub_input.flatten()))
                return np.array(brain_input)
            else:
                # Transform the brain output into a tuple which match the corresponding "nested" Spaces
                brain_output = []
                index = 0
                for sub_space in relevant_space:
                    current_range = self._size_from_space(sub_space)

                    brain_output.append(
                        self.parse_brain_input_output(
                            brain_input_output[index:index + current_range], is_brain_input=False,
                            relevant_space=sub_space
                        )
                    )
                    index += current_range

                assert index == brain_input_output.size

                return tuple(brain_output)
        else:
            raise RuntimeError(
                "The gym.Space '{}' is currently not supported to be parsed.".format(type(relevant_space))
            )

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
