from gym.spaces import Space
import numpy as np
import torch
import torch.nn as nn

from brains.i_layer_based_brain import ILayerBasedBrain, LayerdConfigClass
from brains.i_pytorch_brain import IPytorchBrain
from brains import register_brain
from tools.configurations import LstmLayeredCfg, LstmTorchCfg, IPytorchBrainCfg


@register_brain("LSTM_PyTorch", LstmTorchCfg)
class LSTMPyTorch(IPytorchBrain[LstmTorchCfg]):
    def __init__(self, input_space: Space, output_space: Space,
                 individual: np.ndarray, config: LstmTorchCfg):
        IPytorchBrain.__init__(self, input_space, output_space, individual,
                               config)

    def get_brain(self, config: IPytorchBrainCfg, input_size):
        return nn.LSTM(input_size=input_size,
                       hidden_size=config.hidden_size,
                       num_layers=config.num_layers,
                       bias=config.use_bias)

    @staticmethod
    def get_hidden(config: IPytorchBrainCfg):
        return (torch.randn(config.num_layers, 1, config.hidden_size),
                torch.randn(config.num_layers, 1, config.hidden_size))

    @staticmethod
    def get_number_gates():
        return 4


@register_brain("LSTMLayered", LstmLayeredCfg)
class LSTMLayered(ILayerBasedBrain[LstmLayeredCfg]):
    @staticmethod
    def get_number_hidden_values():
        return 2

    def __init__(self, input_space: Space, output_space: Space,
                 individual: np.ndarray, config: LayerdConfigClass):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def get_number_gates():
        return 4

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h,
                   hidden):
        # Input Gate
        i_t = ILayerBasedBrain.sigmoid(
            np.dot(weight_ih[0], layer_input) + bias_h[0] +
            np.dot(weight_hh[0], hidden[0]))
        # Forget Gate
        f_t = ILayerBasedBrain.sigmoid(
            np.dot(weight_ih[1], layer_input) + bias_h[1] +
            np.dot(weight_hh[1], hidden[0]))
        # Cell Gate
        g_t = np.tanh(
            np.dot(weight_ih[2], layer_input) + bias_h[2] +
            np.dot(weight_hh[2], hidden[0]))
        # Output Gate
        o_t = ILayerBasedBrain.sigmoid(
            np.dot(weight_ih[3], layer_input) + bias_h[3] +
            np.dot(weight_hh[3], hidden[0]))
        cell_state = np.multiply(f_t, hidden[1]) + np.multiply(i_t, g_t)
        hid = np.multiply(o_t, np.tanh(cell_state))
        return [[hid, cell_state], hid]
