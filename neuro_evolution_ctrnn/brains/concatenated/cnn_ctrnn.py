from typing import Union

import numpy as np
from gym.spaces import Space

from brains.brain import Brain, register_brain
from brains.continuous_time_rnn import CTRNN
from brains.cnn import CNN
from tools.configurations import CNNCTRNNCfg


@register_brain("CNN_CTRNN", CNNCTRNNCfg)
class CNNCTRNN(Brain[CNNCTRNNCfg]):
    _usage_key_cnn_output = "_cnn_output_space"

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: CNNCTRNNCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        paramter_usage_dict = self.get_free_parameter_usage(config, input_space, output_space)
        cnn_output_space = paramter_usage_dict[self._usage_key_cnn_output]
        cnn_size = CNN.get_individual_size(config.cnn_conf, input_space, cnn_output_space)
        ctrnn_size = CTRNN.get_individual_size(config.ctrnn_conf, cnn_output_space, output_space)
        ind_cnn = individual[0:cnn_size]
        ind_ctrnn = individual[cnn_size:cnn_size + ctrnn_size]
        self.cnn = CNN(input_space=input_space, output_space=cnn_output_space,
                       config=config.cnn_conf, individual=ind_cnn)
        self.ctrnn = CTRNN(input_space=cnn_output_space, output_space=output_space,
                           config=config.ctrnn_conf, individual=ind_ctrnn)

    def brainstep(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:
        return self.ctrnn.brainstep(self.cnn.brainstep(ob))

    @classmethod
    def get_free_parameter_usage(cls, config: CNNCTRNNCfg, input_space: Space, output_space: Space):
        cnn_output_space = CNN.get_output_shape(config=config.cnn_conf, input_space=input_space)
        cnn_paramter_usage = CNN.get_free_parameter_usage(config=config.cnn_conf, input_space=input_space,
                                                          output_space=cnn_output_space)
        ctrnn_paramter_usage = CTRNN.get_free_parameter_usage(config=config.ctrnn_conf,
                                                              input_space=cnn_output_space,
                                                              output_space=output_space)
        return {"CNN": cnn_paramter_usage, "CTRNN": ctrnn_paramter_usage, cls._usage_key_cnn_output: cnn_output_space}

    @classmethod
    def generate_and_set_class_state(cls, config: CNNCTRNNCfg, input_space: Space, output_space: Space):
        cnn_output_space = CNN.get_output_shape(config=config.cnn_conf, input_space=input_space)
        CTRNN.generate_and_set_class_state(config.ctrnn_conf, cnn_output_space, output_space)

    @classmethod
    def get_class_state(cls):
        return CTRNN.get_class_state()

    @classmethod
    def set_class_state(cls, v_mask, w_mask, t_mask):
        return CTRNN.set_class_state(v_mask, w_mask, t_mask)


