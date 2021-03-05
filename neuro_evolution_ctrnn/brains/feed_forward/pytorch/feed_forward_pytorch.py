from typing import List

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Space

from brains.brain import register_brain
from brains.feed_forward.feed_forward_base import FeedForwardBase
from tools.configurations import FeedForwardCfg


@register_brain("FeedForwardPyTorch", FeedForwardCfg)
class FeedForwardPyTorch(nn.Module, FeedForwardBase[FeedForwardCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: FeedForwardCfg):
        nn.Module.__init__(self)
        FeedForwardBase.__init__(self, input_space, output_space, individual, config)

        # Set activation functions for hidden layers and output layer based on config
        self.activation_hidden_layers = self.get_activation_function_pytorch(config.neuron_activation)
        self.activation_output_layer = self.get_activation_function_pytorch(config.neuron_activation_output)

        self.layers: List[nn.Linear] = []

        index = 0
        previous_layer_size = self.input_size

        # Hidden Layers
        for hidden_layer_size in self.hidden_layers:

            current_layer = nn.Linear(previous_layer_size, hidden_layer_size, bias=self.use_bias)
            current_weight, index = self.read_matrix_from_genome(individual, index, hidden_layer_size, previous_layer_size)
            current_layer.weight.data = torch.from_numpy(current_weight)

            if self.use_bias:
                current_bias, index = self.read_matrix_from_genome(individual, index, hidden_layer_size, 1)
                current_layer.bias.data = torch.from_numpy(current_bias.flatten())

            self.layers.append(current_layer)

            previous_layer_size = hidden_layer_size

        # Output Layer
        self.output_layer: nn.Linear = nn.Linear(previous_layer_size, self.output_size, bias=config.use_bias)
        weights_output_layer, index = self.read_matrix_from_genome(individual, index, self.output_size, previous_layer_size)
        self.output_layer.weight.data = torch.from_numpy(weights_output_layer)

        if self.use_bias:
            output_bias, index = self.read_matrix_from_genome(individual, index, self.output_size, 1)
            self.output_layer.bias.data = torch.from_numpy(output_bias.flatten())

    def brainstep(self, ob: np.ndarray) -> np.ndarray:

        assert ob.ndim == 1
        assert ob.size == self.input_size

        x = ob

        with torch.no_grad():
            x = torch.from_numpy(x.astype(np.float32))

            # Hidden Layers
            for layer in self.layers:
                x = self.activation_hidden_layers(layer(x))

            # Output Layer
            x = self.activation_output_layer(self.output_layer(x))

            return x.view(self.output_size).numpy()
