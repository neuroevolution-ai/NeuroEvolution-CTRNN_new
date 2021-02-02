import torch
import numpy as np
import torch.nn as nn
from brains.i_brain import IBrain
from tools.configurations import FeedForwardCfg
from gym.spaces import Space
from typing import List, Callable


class FeedForward(IBrain[FeedForwardCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config=config, input_space=input_space,
                                                           output_space=output_space)

        self.input_size: int = self._size_from_space(input_space)
        self.output_size: int = self._size_from_space(output_space)
        self.config = config

        # If the check fails the program aborts
        self.hidden_layers: List[int] = self.check_hidden_layers(config.hidden_layers)

        self.non_linearity: Callable[[np.ndarray], np.ndarray]
        if config.non_linearity == "relu":
            if isinstance(self, FeedForwardPyTorch):
                self.non_linearity = nn.ReLU()  # type:ignore
            else:
                self.non_linearity = self.relu
        elif config.non_linearity == "tanh":
            if isinstance(self, FeedForwardPyTorch):
                self.non_linearity = nn.Tanh()  # type:ignore
            else:
                self.non_linearity = self.tanh
        else:
            raise RuntimeError(
                "The chosen non linearity '{}' is not implemented, choose either 'relu' or 'tanh'"
                "".format(config.non_linearity))

    def brainstep(self, ob: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

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

    @classmethod
    def get_individual_size(cls, config: FeedForwardCfg, input_space: Space, output_space: Space) -> int:
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

        return individual_size


class FeedForwardPyTorch(nn.Module, FeedForward):

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

        # TODO this has to be redone if it shall be used
        # # Indirect encoding
        # if config.indirect_encoding:
        #
        #     config_cppn = LayeredNNCfg(number_neurons_layer1=config.cppn_hidden_size1,
        #                                number_neurons_layer2=config.cppn_hidden_size2, indirect_encoding=False,
        #                                use_biases=False, cppn_hidden_size1=0, cppn_hidden_size2=0, type="LNN")
        #
        #     cppn_weights = LayeredNN(4, 1, individual, config_cppn)
        #
        #     self.W1 = np.zeros((self.hidden_size1, self.input_size), dtype=np.single)
        #     self.W2 = np.zeros((self.hidden_size2, self.hidden_size1), dtype=np.single)
        #     self.W3 = np.zeros((self.output_size, self.hidden_size2), dtype=np.single)
        #
        #     for i, j in np.ndindex(self.W1.shape):
        #         self.W1[i, j] = cppn_weights.step(
        #             np.array([i / (self.hidden_size1 - 1), 0.33, j / (self.input_size - 1), 0]))
        #
        #     for i, j in np.ndindex(self.W2.shape):
        #         self.W2[i, j] = cppn_weights.step(
        #             np.array([i / (self.hidden_size2 - 1), 0.66, j / (self.hidden_size1 - 1), 0.33]))
        #
        #     for i, j in np.ndindex(self.W3.shape):
        #         self.W3[i, j] = cppn_weights.step(
        #             np.array([i / (self.output_size - 1), 1.0, j / (self.hidden_size2 - 1), 0.66]))
        #

    def brainstep(self, ob: np.ndarray) -> np.ndarray:
        x = ob

        with torch.no_grad():
            x = torch.from_numpy(x.astype(np.float32))

            for layer in self.layers:
                x = self.non_linearity(layer(x))
            return x.view(self.output_size).numpy()


class FeedForwardNumPy(FeedForward):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        super().__init__(input_space, output_space, individual, config)

        self.weights: [np.ndarray] = []
        self.bias: [np.ndarray] = []

        last_layer = self.input_size
        current_index = 0

        for hidden_layer in self.hidden_layers + [self.output_size]:
            current_size = last_layer * hidden_layer

            current_weight = np.array(individual[current_index:current_index + current_size], dtype=np.single)
            self.weights.append(current_weight.reshape((hidden_layer, last_layer)))

            last_layer = hidden_layer
            current_index += current_size

        if config.use_bias:
            for hidden_layer in self.hidden_layers + [self.output_size]:
                self.bias.append(np.array(individual[current_index:current_index + hidden_layer], dtype=np.single))

                current_index += hidden_layer
        else:
            self.bias = [np.zeros(hidden_layer) for hidden_layer in self.hidden_layers + [self.output_size]]

    @staticmethod
    def layer_step(x: np.ndarray, layer_weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        # If bias is not used it will be zero, see constructor
        return np.dot(layer_weights, x) + bias

    def step(self, ob: np.ndarray) -> np.ndarray:
        x = ob

        for weight, bias in zip(self.weights, self.bias):
            x = self.non_linearity(self.layer_step(x, weight, bias))

        return x
