from abc import ABC

import numpy as np
from tools.configurations import CnnCtrnnCfg, ConvolutionalNNCfg
from typing import List, Union
from gym.spaces import Space, Box
from brains.i_brain import IBrain
from brains.continuous_time_rnn import ContinuousTimeRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


cnn_output_space = Box(-1, 1, (245, 1), np.float16)


# noinspection PyPep8Naming
class CnnCtrnn(IBrain[CnnCtrnnCfg]):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: CnnCtrnnCfg):
        super().__init__(input_space, output_space, individual, config)
        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        self.config = config
        cnn_size, ctrnn_size = self._get_sub_individual_size(config, input_space, output_space)
        ind_cnn = individual[0:cnn_size]
        ind_ctrnn = individual[cnn_size:cnn_size + ctrnn_size]
        self.cnn = Cnn(input_space=input_space, output_space=cnn_output_space, config=config.cnn_conf,
                       individual=ind_cnn)
        self.ctrnn = ContinuousTimeRNN(input_space=cnn_output_space, output_space=output_space,
                                       config=config.ctrnn_conf, individual=ind_ctrnn)

    def step(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:
        # x = torch.from_numpy(ob.astype(np.float32))
        x = torch.from_numpy(np.array([ob])).permute(0, 3, 1, 2)
        cnn_out = self.cnn.forward(x=x)
        return self.ctrnn.step(ob=cnn_out.numpy())

    @classmethod
    def _get_sub_individual_size(cls, config, input_space, output_space):
        cnn_size = Cnn.get_individual_size(config=config.cnn_conf, input_space=input_space,
                                           output_space=cnn_output_space)
        ctrnn_size = ContinuousTimeRNN.get_individual_size(config=config.ctrnn_conf, input_space=cnn_output_space,
                                                           output_space=output_space)
        return cnn_size, ctrnn_size

    @classmethod
    def get_individual_size(cls, config: CnnCtrnnCfg, input_space: Space, output_space: Space):
        a, b = cls._get_sub_individual_size(config, input_space, output_space)
        return a + b

    @classmethod
    def set_masks_globally(cls, config: CnnCtrnnCfg, input_space: Space, output_space: Space):
        ContinuousTimeRNN.set_masks_globally(config.ctrnn_conf, cnn_output_space, output_space)

    @classmethod
    def get_class_state(cls):
        return ContinuousTimeRNN.get_class_state()

    @classmethod
    def set_class_state(cls, v_mask, w_mask, t_mask):
        # https://github.com/pytorch/pytorch/issues/24398
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        logging.info("Setting number of torch-threads to 1")
        # see https://github.com/pytorch/pytorch/issues/13757
        torch.set_num_threads(1)
        return ContinuousTimeRNN.set_class_state(v_mask, w_mask, t_mask)


class Cnn(nn.Module):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConvolutionalNNCfg):
        super().__init__()
        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=(2, 2))
            self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(3, 5, kernel_size=5, stride=1, padding=(2, 2))
            self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2)

            index = 0
            size = self.conv1.weight.size().numel()
            weight = individual[index:index + size]
            self.conv1.weight = nn.Parameter(torch.from_numpy(weight).view(self.conv1.weight.size()).float())
            index = size

            size = self.conv2.weight.size().numel()
            weight = individual[index:index + size]
            self.conv2.weight = nn.Parameter(torch.from_numpy(weight).view(self.conv2.weight.size()).float())


        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = torch.sigmoid(self.conv2(x))
            x = self.maxpool2(x)
            return x.flatten()

    @classmethod
    def get_individual_size(cls, config: ConvolutionalNNCfg, input_space: Space, output_space: Space):
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=(2, 2)).weight.size().numel()
        conv2 = nn.Conv2d(3, 5, kernel_size=5, stride=1, padding=(3, 3)).weight.size().numel()

        return conv1 + conv2
