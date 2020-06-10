import numpy as np

from brains.i_brain import IBrain
from gym.spaces import Space

from i_brain import ConfigClass
from tools.configurations import LSTMCfg


class LSTMNumPy(IBrain):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LSTMCfg):
        super().__init__(input_space, output_space, individual, config)

        self.config = config

        self.input_size = self._size_from_space(input_space)
        self.output_size = self._size_from_space(output_space)
        self.lstm_num_layers = config.lstm_num_layers
        self.use_biases = config.use_biases

        # TODO multiple layers

        self.weight_ih_l0 = np.random.randn(4 * self.output_size, self.input_size).astype(np.float32)
        self.weight_hh_l0 = np.random.randn(4 * self.output_size, self.output_size).astype(np.float32)

        if config.use_biases:
            self.bias_ih_l0 = np.random.randn(4 * self.output_size).astype(np.float32)
            self.bias_hh_l0 = np.random.randn(4 * self.output_size).astype(np.float32)
        else:
            self.bias_ih_l0 = self.bias_hh_l0 = np.zeros(4 * self.output_size).astype(np.float32)

        self.hidden = np.random.randn(self.lstm_num_layers, self.output_size).astype(np.float32)
        self.cell_state = np.random.randn(self.lstm_num_layers, self.output_size).astype(np.float32)

        individual = np.array(individual, dtype=np.float32)

        weight_ih_l0_shape = self.weight_ih_l0.shape
        weight_hh_l0_shape = self.weight_hh_l0.shape

        bias_ih_l0_shape = self.bias_ih_l0.shape
        bias_hh_l0_shape = self.bias_hh_l0.shape

        current_index = 0
        self.weight_ih_l0 = individual[current_index:current_index + self.weight_ih_l0.size].reshape(weight_ih_l0_shape)
        current_index += self.weight_ih_l0.size

        self.weight_hh_l0 = individual[current_index:current_index + self.weight_hh_l0.size].reshape(weight_hh_l0_shape)
        current_index += self.weight_hh_l0.size

        if config.use_biases:
            self.bias_ih_l0 = individual[current_index:current_index + self.bias_ih_l0.size].reshape(bias_ih_l0_shape)
            current_index += self.bias_ih_l0.size

            self.bias_hh_l0 = individual[current_index:current_index + self.bias_hh_l0.size].reshape(bias_hh_l0_shape)
            current_index += self.bias_hh_l0.size

        assert current_index == len(individual)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def step(self, ob: np.ndarray):
        ob = ob.astype(np.float32)

        # Input Gate
        i_t = self.sigmoid(np.dot(self.weight_ih_l0[0:8], ob)
                           + self.bias_ih_l0[0:8]
                           + np.dot(self.weight_hh_l0[0:8], self.hidden[0])
                           + self.bias_hh_l0[0:8])

        f_t = self.sigmoid(np.dot(self.weight_ih_l0[8:16], ob)
                           + self.bias_ih_l0[8:16]
                           + np.dot(self.weight_hh_l0[8:16], self.hidden[0])
                           + self.bias_hh_l0[8:16])

        g_t = np.tanh(np.dot(self.weight_ih_l0[16:24], ob)
                      + self.bias_ih_l0[16:24]
                      + np.dot(self.weight_hh_l0[16:24], self.hidden[0])
                      + self.bias_hh_l0[16:24])

        o_t = self.sigmoid(np.dot(self.weight_ih_l0[24:32], ob)
                           + self.bias_ih_l0[24:32]
                           + np.dot(self.weight_hh_l0[24:32], self.hidden[0])
                           + self.bias_hh_l0[24:32])

        self.cell_state[0] = np.multiply(f_t, self.cell_state[0]) + np.multiply(i_t, g_t)
        self.hidden[0] = np.multiply(o_t, np.tanh(self.cell_state[0]))

        return np.copy(self.hidden[0])

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        lstm_num_layers = config.lstm_num_layers

        individual_size = 0

        # Calculate the number of weights as depicted in https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        if lstm_num_layers > 0:
            individual_size += 4 * output_size * (input_size + output_size)

            if config.use_biases:
                individual_size += 8 * output_size

        for i in range(1, lstm_num_layers):
            # Here it is assumed that the LSTM is not bidirectional
            individual_size += 8 * output_size * output_size

            if config.use_biases:
                individual_size += 8 * output_size

        return individual_size
