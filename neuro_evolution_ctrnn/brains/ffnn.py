import torch
import numpy as np
import torch.nn as nn
from brains.i_brain import IBrain, register_brain
from tools.configurations import FeedForwardCfg
from gym.spaces import Space
from typing import List, Callable, TypeVar

FeedForwardConfigClass = TypeVar('FeedForwardConfigClass', bound=FeedForwardCfg)


class FeedForward(IBrain[FeedForwardConfigClass]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config=config, input_space=input_space,
                                                           output_space=output_space)

        self.input_size: int = self._size_from_space(input_space)
        self.output_size: int = self._size_from_space(output_space)
        self.config = config

        # If the check fails the program aborts
        self.hidden_layers: List[int] = self.check_hidden_layers(config.hidden_layers)

        # Set activation functions for hidden layers and output layer based on config
        self.neuron_activation_hidden_layers = self.__get_neuron_activation(config.neuron_activation_hidden_layers)
        self.neuron_activation_output_layer = self.__get_neuron_activation(config.neuron_activation_output_layer)

    def __get_neuron_activation(self, activation: str) -> Callable[[np.ndarray], np.ndarray]:

        if activation == "relu":
            if isinstance(self, FeedForwardPyTorch):
                return nn.ReLU()  # type:ignore
            else:
                return self.relu
        elif activation == "linear":
            if isinstance(self, FeedForwardPyTorch):
                return nn.Linear()  # type:ignore
            else:
                return self.linear
        elif activation == "tanh":
            if isinstance(self, FeedForwardPyTorch):
                return nn.Tanh()  # type:ignore
            else:
                return self.tanh
        else:
            raise RuntimeError(
                "The chosen non linearity '{}' is not implemented, choose either 'relu' or 'tanh'"
                "".format(activation))

    def brainstep(self, ob: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def check_hidden_layers(hidden_layers: List[int]) -> List[int]:
        try:
            assert len(hidden_layers) > 0

            # Design Space, unpack first
            if any(isinstance(i, list) for i in hidden_layers):
                # Simply take the first list, should be the only one. If not somewhere the decision of the Design Space
                # did not occur
                hidden_layers = hidden_layers[0]
            assert all(hidden_layers) > 0
            return hidden_layers
        except AssertionError:
            raise RuntimeError(
                "Error with the chosen hidden layer list {}. It must be at least of size 1 and have elements which are "
                "larger than 0.".format(hidden_layers))

    @staticmethod
    def read_from_genome(individual: np.ndarray, index: int, matrix_rows: int, matrix_columns: int):
        matrix_size = matrix_columns * matrix_rows

        matrix = np.array(individual[index:index + matrix_size], dtype=np.single)
        matrix = matrix.reshape(matrix_rows, matrix_columns)

        index += matrix_size

        return matrix, index

    @classmethod
    def get_free_parameter_usage(cls, config: FeedForwardConfigClass, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        hidden_layers = cls.check_hidden_layers(config.hidden_layers)

        individual_size = 0
        last_layer = input_size

        for hidden_layer in hidden_layers:
            individual_size += last_layer * hidden_layer
            last_layer = hidden_layer

        individual_size += last_layer * output_size

        if config.use_bias:
            individual_size += sum(hidden_layers) + output_size

        return {'individual_size': individual_size}


@register_brain("FeedForward_PyTorch", FeedForwardCfg)
class FeedForwardPyTorch(nn.Module, FeedForward[FeedForwardCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        nn.Module.__init__(self)
        FeedForward.__init__(self, input_space, output_space, individual, config)

        self.layers: [nn.Linear] = []

        current_index = 0
        last_layer = self.input_size
        for hidden_layer in self.hidden_layers + [self.output_size]:
            current_size = last_layer * hidden_layer

            next_layer = nn.Linear(last_layer, hidden_layer, bias=config.use_bias)
            next_layer.weight.data = torch.from_numpy(
                np.array(individual[current_index:current_index + current_size], dtype=np.single).reshape(
                    (hidden_layer, last_layer)))

            self.layers.append(next_layer)
            last_layer = hidden_layer
            current_index += current_size

        if config.use_bias:
            # Take extra loop for bias because we have the "convention" to add bias to the end of the individual array
            for index, hidden_layer in enumerate(self.hidden_layers + [self.output_size]):
                self.layers[index].bias.data = torch.from_numpy(
                    np.array(individual[current_index:current_index + hidden_layer]))
                current_index += hidden_layer

    def step(self, ob: np.ndarray) -> np.ndarray:

        x = ob

        with torch.no_grad():
            x = torch.from_numpy(x.astype(np.float32))

            for layer in self.layers:
                x = self.neuron_activation_hidden_layers(layer(x))
            return x.view(self.output_size).numpy()


@register_brain("FeedForward_NumPy", FeedForwardCfg)
class FeedForwardNumPy(FeedForward[FeedForwardCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        super().__init__(input_space, output_space, individual, config)

        self.weights: List[np.ndarray] = []
        self.bias: List[np.ndarray] = []
        self.use_bias = config.use_bias

        previous_layer = self.input_size
        index = 0

        for hidden_layer in self.hidden_layers + [self.output_size]:
            current_weight, index = self.read_from_genome(individual, index, hidden_layer, previous_layer)
            self.weights.append(current_weight)
            previous_layer = hidden_layer

        if self.use_bias:
            for hidden_layer in self.hidden_layers + [self.output_size]:
                current_bias, index = self.read_from_genome(individual, index, hidden_layer, 1)
                self.bias.append(current_bias)

    def step(self, ob: np.ndarray) -> np.ndarray:
        x = ob

        if self.use_bias:
            for weight, bias in zip(self.weights, self.bias):
                x = np.matmul(weight, x)
                x = np.add(x, bias)
                x = self.neuron_activation_hidden_layers(x)
        else:
            for weight in self.weights:
                x = np.matmul(weight, x)
                x = self.neuron_activation_hidden_layers(x)

        return x
