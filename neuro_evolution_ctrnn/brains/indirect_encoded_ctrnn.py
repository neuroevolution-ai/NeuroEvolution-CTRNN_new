import numpy as np
from tools.configurations import CnnCtrnnCfg, ConvolutionalNNCfg
from typing import List, Union
from gym.spaces import Space, Box, flatdim
from brains import IBrain, register_brain
from brains.continuous_time_rnn import ContinuousTimeRNN
from brains.ffnn import FeedForward
from tools.configurations import IndirectEncodedCtrnnCfg
import logging


# noinspection PyPep8Naming
@register_brain("Indirect_CTRNN", IndirectEncodedCtrnnCfg)
class IndirectEncodedCtrnn(IBrain[IndirectEncodedCtrnnCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: IndirectEncodedCtrnnCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        self.config = config

        cppn_input_space = Box(-1, 1, (6,))
        cppn_output_space = Box(-1, 1, (1,))

        self.cppn = FeedForward(input_space=cppn_input_space, output_space=cppn_output_space, config=config.ffnn_conf, individual=individual)

        individual_size_ctrnn = ContinuousTimeRNN.get_individual_size(config.ctrnn_conf, input_space, output_space)
        individual_ctrnn = np.zeros(individual_size_ctrnn)

        self.ctrnn = ContinuousTimeRNN(input_space=input_space, output_space=output_space, config=config.ctrnn_conf, individual=individual_ctrnn)

        ## Set CTRNN matrizes

    def brainstep(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:

        ## Preprocess input

        return self.ctrnn.brainstep(ob=ob)


    @classmethod
    def get_free_parameter_usage(cls, config: IndirectEncodedCtrnnCfg, input_space: Space, output_space: Space):

        cppn_input_space = Box(-1, 1, (6,))
        cppn_output_space = Box(-1, 1, (1,))

        individual_size = FeedForward.get_individual_size(config.ffnn_conf, cppn_input_space, cppn_output_space)

        return {'individual_size': individual_size}

    @classmethod
    def generate_and_set_class_state(cls, config: IndirectEncodedCtrnnCfg, input_space: Space, output_space: Space):
        ContinuousTimeRNN.generate_and_set_class_state(config=config.ctrnn_conf, input_space=input_space, output_space=output_space)
