import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Space

from brains.brain import register_brain
from brains.recurrent.pytorch.recurrent_based_brain_pytorch import RecurrentBasedBrainPyTorch
from tools.configurations import GRUPyTorchCfg, RecurrentBasedBrainPyTorchCfg


@register_brain("GRUPyTorch", GRUPyTorchCfg)
class GRUPyTorch(RecurrentBasedBrainPyTorch[GRUPyTorchCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: GRUPyTorchCfg):
        RecurrentBasedBrainPyTorch.__init__(self, input_space, output_space, individual, config)

    def get_brain(self, config: RecurrentBasedBrainPyTorchCfg, input_size):
        return nn.GRU(
            input_size=input_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, bias=config.use_bias)

    @staticmethod
    def get_hidden(config: RecurrentBasedBrainPyTorchCfg):
        return (
            torch.randn(config.num_layers, 1, config.hidden_size)
        )

    @staticmethod
    def get_number_gates():
        return 3
