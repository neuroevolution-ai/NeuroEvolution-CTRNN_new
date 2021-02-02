import abc
import torch
from gym import Space
from torch import nn
import numpy as np
from brains.i_brain import IBrain, ConfigClass
from tools.configurations import IPytorchBrainCfg


class IPytorchBrain(nn.Module, IBrain, abc.ABC):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: IPytorchBrainCfg):
        nn.Module.__init__(self)
        IBrain.__init__(self, input_space, output_space, individual, config)
        self.output_size = IBrain._size_from_space(self.output_space)
        assert len(individual) == self.get_individual_size(
            config=config, input_space=input_space, output_space=output_space)

        if config.num_layers <= 0:
            raise RuntimeError("PyTorch need at least one layer.")

        # Disable tracking of the gradients since backpropagation is not used
        with torch.no_grad():
            self.brain = self.get_brain(config, IBrain._size_from_space(input_space))

            # Iterate through all layers and assign the weights from the individual
            current_index = 0
            for i in range(config.num_layers):
                attr_weight_ih_li = "weight_ih_l{}".format(i)
                attr_weight_hh_li = "weight_hh_l{}".format(i)

                weight_ih_li = getattr(self.brain, attr_weight_ih_li)
                weight_hh_li = getattr(self.brain, attr_weight_hh_li)

                weight_ih_li_size = np.prod(weight_ih_li.size())
                weight_hh_li_size = np.prod(weight_hh_li.size())

                # Do not forget to reshape back again
                weight_ih_li.data = torch.from_numpy(
                    individual[current_index: current_index + weight_ih_li_size]).view(weight_ih_li.size()).float()
                current_index += weight_ih_li_size

                weight_hh_li.data = torch.from_numpy(
                    individual[current_index: current_index + weight_hh_li_size]).view(weight_hh_li.size()).float()
                current_index += weight_hh_li_size

                if config.use_bias:
                    attr_bias_ih_li = "bias_ih_l{}".format(i)
                    attr_bias_hh_li = "bias_hh_l{}".format(i)

                    bias_ih_li = getattr(self.brain, attr_bias_ih_li)
                    bias_hh_li = getattr(self.brain, attr_bias_hh_li)

                    bias_ih_li_size = bias_ih_li.size()[0]
                    bias_hh_li_size = bias_hh_li.size()[0]

                    bias_ih_li.data = torch.from_numpy(
                        individual[current_index: current_index + bias_ih_li_size]).float()
                    current_index += bias_ih_li_size

                    bias_hh_li.data = torch.from_numpy(
                        individual[current_index: current_index + bias_hh_li_size]).float()
                    current_index += bias_hh_li_size
            self.weight_ho = np.array(
                individual[current_index: current_index + self.output_size * config.hidden_size]).reshape(
                (self.output_size, config.hidden_size))
            current_index += self.output_size * config.hidden_size
            assert current_index == len(individual)

            self.hidden = self.get_hidden(config)

    @classmethod
    def get_individual_size(cls, config: ConfigClass, input_space: Space, output_space: Space):
        num_layers = config.num_layers
        number_gates = cls.get_number_gates()
        hidden_size = config.hidden_size
        index = 0
        # size of the learnable input-hidden weights

        index += hidden_size * IBrain._size_from_space(input_space) * number_gates
        if num_layers > 1:
            index += hidden_size * hidden_size * (num_layers - 1) * number_gates

        # size of the learnable hidden-hidden weights
        index += hidden_size * hidden_size * num_layers * number_gates

        if config.use_bias:
            index += 2 * hidden_size * num_layers * number_gates
        index += IBrain._size_from_space(output_space) * hidden_size
        return index

    @staticmethod
    @abc.abstractmethod
    def get_number_gates():
        pass

    @abc.abstractmethod
    def get_brain(self, config: IPytorchBrainCfg, input_size):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_hidden(config: IPytorchBrainCfg):
        pass

    def brainstep(self, ob: np.ndarray):

        with torch.no_grad():
            # Input requires the form (seq_len, batch, input_size)
            out, self.hidden = self.brain(torch.from_numpy(ob.astype(np.float32)).view(1, 1, -1), self.hidden)
            return np.dot(self.weight_ho, out.flatten())
