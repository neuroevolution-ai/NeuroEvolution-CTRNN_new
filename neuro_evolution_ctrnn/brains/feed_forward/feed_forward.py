import itertools
from typing import List

import numpy as np
from gym.spaces import Space

from brains.brain import register_brain
from brains.feed_forward.feed_forward_base import FeedForwardBase
from tools.configurations import FeedForwardCfg


@register_brain("FeedForward", FeedForwardCfg)
class FeedForward(FeedForwardBase[FeedForwardCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        super().__init__(input_space, output_space, individual, config)

        # Set activation functions for hidden layers and output layer based on config
        self.activation_hidden_layers = self.get_activation_function_numpy(config.neuron_activation)
        self.activation_output_layer = self.get_activation_function_numpy(config.neuron_activation_output)

        self.weights_hidden_layers: List[np.ndarray] = []
        self.biases_hidden_layers: List[np.ndarray] = []

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

    def brainstep(self, ob: np.ndarray) -> np.ndarray:

        assert ob.ndim == 1
        assert ob.size == self.input_size

        x = self.predict(ob.reshape(ob.size, 1))

        return x.flatten()

    def predict(self, ob: np.ndarray) -> np.ndarray:

        assert ob.ndim > 1
        assert ob.shape[-2] == self.input_size
        assert ob.shape[-1] == 1

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
