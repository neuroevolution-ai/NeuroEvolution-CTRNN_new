import numpy as np
from gym.spaces import Space

from brains.brain import register_brain
from recurrent.recurrent_based_brain import RecurrentBasedBrain
from tools.configurations import LSTMCfg


@register_brain("LSTM", LSTMCfg)
class LSTM(RecurrentBasedBrain[LSTMCfg]):

    @staticmethod
    def get_number_hidden_values():
        return 2

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LSTMCfg):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def get_number_gates():
        return 4

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        # Input Gate
        i_t = RecurrentBasedBrain.sigmoid(np.dot(weight_ih[0], layer_input)
                                          + bias_h[0]
                                          + np.dot(weight_hh[0], hidden[0]))
        # Forget Gate
        f_t = RecurrentBasedBrain.sigmoid(np.dot(weight_ih[1], layer_input)
                                          + bias_h[1]
                                          + np.dot(weight_hh[1], hidden[0]))
        # Cell Gate
        g_t = np.tanh(np.dot(weight_ih[2], layer_input)
                      + bias_h[2]
                      + np.dot(weight_hh[2], hidden[0]))
        # Output Gate
        o_t = RecurrentBasedBrain.sigmoid(np.dot(weight_ih[3], layer_input)
                                          + bias_h[3]
                                          + np.dot(weight_hh[3], hidden[0]))
        cell_state = np.multiply(f_t, hidden[1]) + np.multiply(i_t, g_t)
        hid = np.multiply(o_t, np.tanh(cell_state))
        return [[hid, cell_state], hid]
