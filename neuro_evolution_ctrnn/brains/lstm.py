from gym.spaces import Space
import numpy as np
import torch
import torch.nn as nn

from brains.i_brain import IBrain
from brains.i_layer_based_brain import ILayerBasedBrain, LayerdConfigClass
from brains.i_pytorch_brain import IPytorchBrain
from tools.configurations import LSTMCfg, LstmLayeredCfg, LstmTorchCfg, IPytorchBrainCfg


class LSTM(IBrain):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LSTMCfg):
        super().__init__(input_space, output_space, individual, config)

        self.config = config
        self.input_space = input_space

        self.input_size = self._size_from_space(input_space)
        self.output_size = self._size_from_space(output_space)
        self.lstm_num_layers = config.lstm_num_layers

    def brainstep(self, ob: np.ndarray):
        pass

    @classmethod
    def get_individual_size(cls, config: LSTMCfg, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        lstm_num_layers = config.lstm_num_layers

        individual_size = 0

        # Calculate the number of weights as depicted in https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        if lstm_num_layers > 0:
            individual_size += 4 * output_size * (input_size + output_size)

            if config.use_bias:
                individual_size += 8 * output_size

        for i in range(1, lstm_num_layers):
            # Here it is assumed that the LSTM is not bidirectional
            individual_size += 8 * output_size * output_size

            if config.use_bias:
                individual_size += 8 * output_size

        return individual_size

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class LSTMPyTorch(IPytorchBrain):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LstmTorchCfg):
        IPytorchBrain.__init__(self, input_space, output_space, individual, config)

    def get_brain(self, config: IPytorchBrainCfg, input_size):
        return nn.LSTM(
            input_size=input_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, bias=config.use_bias)

    @staticmethod
    def get_hidden(config: IPytorchBrainCfg):
        return (
            torch.randn(config.num_layers, 1, config.hidden_size),
            torch.randn(config.num_layers, 1, config.hidden_size)
        )

    @staticmethod
    def get_number_gates():
        return 4

class LSTMNumPy(LSTM):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LSTMCfg):
        super().__init__(input_space, output_space, individual, config)

        if self.lstm_num_layers <= 0:
            raise RuntimeError("LSTMs need at least one layer.")

        # Initialize the first layer, shape will be used for following layers
        # First dimension is 4 because it represents the weights for each of the four gates (input, forget, cell, output
        # gate)
        self.weight_ih_l0 = np.random.randn(4, self.output_size, self.input_size).astype(np.float32)
        self.weight_hh_l0 = np.random.randn(4, self.output_size, self.output_size).astype(np.float32)

        if config.use_bias:
            self.bias_ih_l0 = np.random.randn(4, self.output_size).astype(np.float32)
            self.bias_hh_l0 = np.random.randn(4, self.output_size).astype(np.float32)
        else:
            self.bias_ih_l0 = self.bias_hh_l0 = np.zeros((4, self.output_size)).astype(np.float32)

        self.hidden = np.random.randn(self.lstm_num_layers, self.output_size).astype(np.float32)
        self.cell_state = np.random.randn(self.lstm_num_layers, self.output_size).astype(np.float32)

        individual = np.array(individual, dtype=np.float32)

        current_index = 0

        if self.lstm_num_layers > 0:

            self.weight_ih_l0 = individual[current_index:current_index + self.weight_ih_l0.size].reshape(
                self.weight_ih_l0.shape)
            current_index += self.weight_ih_l0.size

            self.weight_hh_l0 = individual[current_index:current_index + self.weight_hh_l0.size].reshape(
                self.weight_hh_l0.shape)
            current_index += self.weight_hh_l0.size

            if config.use_bias:
                self.bias_ih_l0 = individual[current_index:current_index + self.bias_ih_l0.size].reshape(
                    self.bias_ih_l0.shape)
                current_index += self.bias_ih_l0.size

                self.bias_hh_l0 = individual[current_index:current_index + self.bias_hh_l0.size].reshape(
                    self.bias_hh_l0.shape)
                current_index += self.bias_hh_l0.size

        weight_shape = (4, self.output_size, self.output_size)
        bias_shape = (4, self.output_size)

        weight_size = np.prod(weight_shape)
        bias_size = np.prod(bias_shape)

        # Weights for following layers have not been created
        for i in range(1, self.lstm_num_layers):

            setattr(
                self,
                "weight_ih_l{}".format(i),
                individual[current_index:current_index + weight_size].reshape(weight_shape))
            current_index += weight_size

            setattr(
                self,
                "weight_hh_l{}".format(i),
                individual[current_index:current_index + weight_size].reshape(weight_shape))
            current_index += weight_size

            attr_bias_ih_li = "bias_ih_l{}".format(i)
            attr_bias_hh_li = "bias_hh_l{}".format(i)

            if config.use_bias:
                setattr(self, attr_bias_ih_li, individual[current_index:current_index + bias_size].reshape(bias_shape))
                current_index += bias_size

                setattr(self, attr_bias_hh_li, individual[current_index:current_index + bias_size].reshape(bias_shape))
                current_index += bias_size
            else:
                # Initialize all not used biases with zeros, since they only get added in a step
                # Therefore remove the need to check for biases every time when a prediction is called
                setattr(self, attr_bias_ih_li, np.zeros(bias_shape).astype(np.float32))
                setattr(self, attr_bias_hh_li, np.zeros(bias_shape).astype(np.float32))

        assert current_index == len(individual)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def brainstep(self, ob: np.ndarray):

        x = ob.astype(np.float32)

        # The input for the i-th layer is the (i-1)-th hidden feature or if i==0 the input
        # Calculated as in the PyTorch description of the LSTM:
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        for i in range(self.lstm_num_layers):
            weight_ih_li = getattr(self, "weight_ih_l{}".format(i))
            weight_hh_li = getattr(self, "weight_hh_l{}".format(i))

            # Even if bias is not used they got initialized (to zeros in this case)
            bias_ih_li = getattr(self, "bias_ih_l{}".format(i))
            bias_hh_li = getattr(self, "bias_hh_l{}".format(i))

            # Input Gate
            i_t = self.sigmoid(np.dot(weight_ih_li[0], x)
                               + bias_ih_li[0]
                               + np.dot(weight_hh_li[0], self.hidden[i])
                               + bias_hh_li[0])

            # Forget Gate
            f_t = self.sigmoid(np.dot(weight_ih_li[1], x)
                               + bias_ih_li[1]
                               + np.dot(weight_hh_li[1], self.hidden[i])
                               + bias_hh_li[1])

            # Cell Gate
            g_t = np.tanh(np.dot(weight_ih_li[2], x)
                          + bias_ih_li[2]
                          + np.dot(weight_hh_li[2], self.hidden[i])
                          + bias_hh_li[2])

            # Output Gate
            o_t = self.sigmoid(np.dot(weight_ih_li[3], x)
                               + bias_ih_li[3]
                               + np.dot(weight_hh_li[3], self.hidden[i])
                               + bias_hh_li[3])

            self.cell_state[i] = np.multiply(f_t, self.cell_state[i]) + np.multiply(i_t, g_t)
            self.hidden[i] = np.multiply(o_t, np.tanh(self.cell_state[i]))

            x = self.hidden[i]

        return np.copy(self.hidden[-1])


class LSTMLayered(ILayerBasedBrain[LstmLayeredCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LayerdConfigClass):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def get_number_gates():
        return 4

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        # Input Gate
        i_t = ILayerBasedBrain.sigmoid(np.dot(weight_ih[0], layer_input)
                                       + bias_h[0]
                                       + np.dot(weight_hh[0], hidden))

        # Forget Gate
        f_t = ILayerBasedBrain.sigmoid(np.dot(weight_ih[1], layer_input)
                                       + bias_h[1]
                                       + np.dot(weight_hh[1], hidden))

        # Cell Gate
        g_t = np.tanh(np.dot(weight_ih[2], layer_input)
                      + bias_h[2]
                      + np.dot(weight_hh[2], hidden))

        # Output Gate
        o_t = ILayerBasedBrain.sigmoid(np.dot(weight_ih[3], layer_input)
                                       + bias_h[3]
                                       + np.dot(weight_hh[3], hidden))

        cell_state = np.multiply(f_t, hidden) + np.multiply(i_t, g_t)
        output = np.multiply(o_t, np.tanh(cell_state))
        return [cell_state, output]