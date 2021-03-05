import numpy as np
from gym import Space

from brains.brain import register_brain
from brains.recurrent.recurrent_based_brain import RecurrentBasedBrain
from tools.configurations import GRUCfg


@register_brain("GRU", GRUCfg)
class GRU(RecurrentBasedBrain[GRUCfg]):

    @staticmethod
    def get_number_hidden_values():
        return 1

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: GRUCfg):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def layer_step(layer_input, weight_ih, weight_hh, bias_h, hidden):
        # Reset Gate
        r_t = RecurrentBasedBrain.sigmoid(np.dot(weight_ih[0], layer_input)
                                          + np.dot(weight_hh[0], hidden[0])
                                          + bias_h[0])

        # Update Gate
        z_t = RecurrentBasedBrain.sigmoid(np.dot(weight_ih[1], layer_input)
                                          + np.dot(weight_hh[1], hidden[0])
                                          + bias_h[1])

        # New Gate
        n_t = np.tanh(np.dot(weight_ih[2], layer_input)
                      + np.dot(r_t, np.dot(weight_hh[2], hidden[0]))
                      + bias_h[2])

        result = np.multiply(1 - z_t, n_t) + np.multiply(z_t, hidden[0])
        return [[result], result]

    @staticmethod
    def get_number_gates():
        return 3
