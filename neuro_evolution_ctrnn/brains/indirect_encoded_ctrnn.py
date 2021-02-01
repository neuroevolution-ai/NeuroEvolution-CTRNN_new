from abc import ABC

import numpy as np
from tools.configurations import CnnCtrnnCfg, ConvolutionalNNCfg
from typing import List, Union
from gym.spaces import Space, Box
from brains.i_brain import IBrain
from brains.continuous_time_rnn import ContinuousTimeRNN
import logging


# noinspection PyPep8Naming
class IndirectEncodedCtrnn(IBrain[IndirectEncodedCtrnnCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: CnnCtrnnCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        self.config = config
        cnn_size, ctrnn_size, cnn_output_space = self._get_sub_individual_size(config, input_space, output_space)
        ind_cnn = individual[0:cnn_size]
        ind_ctrnn = individual[cnn_size:cnn_size + ctrnn_size]
        self.cnn = Cnn(input_space=input_space, output_space=cnn_output_space, config=config.cnn_conf,
                       individual=ind_cnn)
        self.ctrnn = ContinuousTimeRNN(input_space=cnn_output_space, output_space=output_space,
                                       config=config.ctrnn_conf, individual=ind_ctrnn)

    def step(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:
        return self.ctrnn.step(ob=ob)

    @classmethod
    def get_individual_size(cls, config: CnnCtrnnCfg, input_space: Space, output_space: Space):
        a, b, _ = cls._get_sub_individual_size(config, input_space, output_space)
        return a + b

    @classmethod
    def generate_and_set_class_state(cls, config: CnnCtrnnCfg, input_space: Space, output_space: Space):
        cnn_output_space = Cnn.get_output_shape(config=config.cnn_conf, input_space=input_space)
        ContinuousTimeRNN.generate_and_set_class_state(config.ctrnn_conf, cnn_output_space, output_space)

    @classmethod
    def get_class_state(cls):
        return ContinuousTimeRNN.get_class_state()

    @classmethod
    def set_class_state(cls, v_mask, w_mask, t_mask):
        return ContinuousTimeRNN.set_class_state(v_mask, w_mask, t_mask)
