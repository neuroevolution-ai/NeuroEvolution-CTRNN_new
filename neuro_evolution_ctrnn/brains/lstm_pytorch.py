import numpy as np
import torch
import torch.nn as nn
from gym import Space

from brains.i_brain import IBrain
from i_brain import ConfigClass
from tools.configurations import LSTMCfg


class LSTMPyTorch(nn.Module, IBrain):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LSTMCfg):
        nn.Module.__init__(self)
        IBrain.__init__(self, input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(
            config=config, input_space=input_space, output_space=output_space)

        self.config = config

        self.input_size = self._size_from_space(input_space)
        self.output_size = self._size_from_space(output_space)
        self.lstm_num_layers = config.brain.lstm_num_layers
        self.use_biases = config.brain.use_biases

        # Disable tracking of the gradients since backpropagation is not used
        with torch.no_grad():
            self.lstm = nn.LSTM(self.input_size, self.output_size, num_layers=self.lstm_num_layers, bias=self.use_biases)

            # Iterate through all layers and assign the weights from the individual
            current_index = 0
            for i in range(self.lstm_num_layers):
                attr_weight_ih_li = "weight_ih_l{}".format(i)
                attr_weight_hh_li = "weight_hh_l{}".format(i)

                weight_ih_li = getattr(self.lstm, attr_weight_ih_li)
                weight_hh_li = getattr(self.lstm, attr_weight_hh_li)

                weight_ih_li_size = np.prod(weight_ih_li.size())
                weight_hh_li_size = np.prod(weight_hh_li.size())

                # Do not forget to reshape back again
                weight_ih_li.data = torch.from_numpy(
                    individual[current_index: current_index + weight_ih_li_size]).view(weight_ih_li.size())
                current_index += weight_ih_li_size

                weight_hh_li.data = torch.from_numpy(
                    individual[current_index: current_index + weight_hh_li_size]).view(weight_hh_li.size())
                current_index += weight_hh_li_size

                if self.use_biases:
                    attr_bias_ih_li = "bias_ih_l{}".format(i)
                    attr_bias_hh_li = "bias_hh_l{}".format(i)

                    bias_ih_li = getattr(self.lstm, attr_bias_ih_li)
                    bias_hh_li = getattr(self.lstm, attr_bias_hh_li)

                    bias_ih_li_size = bias_ih_li.size()[0]
                    bias_hh_li_size = bias_hh_li.size()[0]

                    bias_ih_li.data = torch.from_numpy(individual[current_index: current_index + bias_ih_li_size])
                    current_index += bias_ih_li_size

                    bias_hh_li.data = torch.from_numpy(individual[current_index: current_index + bias_hh_li_size])
                    current_index += bias_hh_li_size

            assert current_index == len(individual)

            # TODO Maybe the hidden values can be initialized differently
            self.hidden = (
                torch.randn(self.lstm_num_layers, 1, self.output_size),
                torch.randn(self.lstm_num_layers, 1, self.output_size)
            )

    def step(self, ob: np.ndarray):
        with torch.no_grad():
            # Input requires the form (seq_len, batch, input_size)
            out, self.hidden = self.lstm(torch.from_numpy(ob).view(1, 1, -1), self.hidden)
            return out.view(self.output_size).numpy()

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        lstm_num_layers = config.brain.lstm_num_layers

        individual_size = 0

        # Calculate the number of weights as depicted in https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        if lstm_num_layers > 0:
            individual_size += 4 * output_size * (input_size + output_size)

            if config.brain.use_biases:
                individual_size += 8 * output_size

        for i in range(1, lstm_num_layers):
            # Here it is assumed that the LSTM is not bidirectional
            individual_size += 8 * output_size * output_size

            if config.brain.use_biases:
                individual_size += 8 * output_size

        return individual_size
