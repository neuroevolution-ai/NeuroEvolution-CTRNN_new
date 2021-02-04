import torch
import numpy as np
import itertools
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

    def step(self, ob: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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

        # Set activation functions for hidden layers and output layer based on config
        self.activation_hidden_layers = self.get_activation_function_pytorch(config.neuron_activation)
        self.activation_output_layer = self.get_activation_function_pytorch(config.neuron_activation_output)

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

        # Set activation functions for hidden layers and output layer based on config
        self.activation_hidden_layers = self.get_activation_function_numpy(config.neuron_activation)
        self.activation_output_layer = self.get_activation_function_numpy(config.neuron_activation_output)

        self.weights_hidden_layers: List[np.ndarray] = []
        self.biases_hidden_layers: List[np.ndarray] = []
        self.use_bias = config.use_bias

        index = 0
        previous_layer_size = self.input_size

        # Read out weight matrizes and bias matrizes from genome for hidden layers
        for hidden_layer in self.hidden_layers:
            current_weight, index = self.read_matrix_from_genome(individual, index, hidden_layer, previous_layer_size)
            self.weights_hidden_layers.append(current_weight)

            if self.use_bias:
                current_bias, index = self.read_matrix_from_genome(individual, index, hidden_layer, 1)
                self.biases_hidden_layers.append(current_bias)

            previous_layer_size = hidden_layer

        # Read out weight matrix from genome for output layer
        self.weights_output_layer, index = self.read_matrix_from_genome(individual, index, self.output_size,
                                                                        previous_layer_size)

        # Read out bias matrix from genome for output layer
        if self.use_bias:
            self.biases_output_layer, index = self.read_matrix_from_genome(individual, index, self.output_size, 1)

    def step(self, ob: np.ndarray) -> np.ndarray:
        x = ob

        # Calculate hidden layers
        # Note: itertools.zip_longest is necessary here instead of zip, since self.biases_hidden_layers is
        # an empty list, if self.use_bias is False
        for weights, biases in itertools.zip_longest(self.weights_hidden_layers, self.biases_hidden_layers):
            x = np.matmul(weights, x)
            if self.use_bias:
                x = np.add(x, biases)
            x = self.activation_hidden_layers(x)

        # Calculate output layer
        x = np.matmul(self.weights_output_layer, x)
        if self.use_bias:
            x = np.add(x, self.biases_output_layer)
        x = self.activation_output_layer(x)

        return x
