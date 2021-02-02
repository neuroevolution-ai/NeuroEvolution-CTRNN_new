import numpy as np
import torch
from gym import Space
from torch import nn

from brains.i_brain import IBrain, ConfigClass
from brains.i_pytorch_brain import IPytorchBrain
from tools.configurations import GruCfg, IPytorchBrainCfg, GruTorchCfg
from brains.i_layer_based_brain import ILayerBasedBrain


class GruNetwork(ILayerBasedBrain[GruCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def layer_step(layer_input, weight_ih, weight_hh, bias_h, hidden):
        # Reset Gate
        r_t = ILayerBasedBrain.sigmoid(np.dot(weight_ih[0], layer_input)
                                       + np.dot(weight_hh[0], hidden)
                                       + bias_h[0])

        # Update Gate
        z_t = ILayerBasedBrain.sigmoid(np.dot(weight_ih[1], layer_input)
                                       + np.dot(weight_hh[1], hidden)
                                       + bias_h[1])

        # New Gate
        n_t = np.tanh(np.dot(weight_ih[2], layer_input)
                      + np.dot(r_t, np.dot(weight_hh[2], hidden))
                      + bias_h[2])

        result = np.multiply(1 - z_t, n_t) + np.multiply(z_t, hidden)
        return [result, result]

    @staticmethod
    def get_number_gates():
        return 3


class GruPyTorch(IPytorchBrain):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: GruTorchCfg):
        IPytorchBrain.__init__(self, input_space, output_space, individual, config)

    def get_brain(self, config: IPytorchBrainCfg, input_size):
        return nn.GRU(
            input_size=input_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, bias=config.use_bias)

    @staticmethod
    def get_hidden(config: IPytorchBrainCfg):
        return (
            torch.randn(config.num_layers, 1, config.hidden_size)
        )

    @staticmethod
    def get_number_gates():
        return 3
