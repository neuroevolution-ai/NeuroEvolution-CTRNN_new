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


# noinspection PyPep8Naming
class CnnCtrnn(IBrain[CnnCtrnnCfg]):
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

    def brainstep(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:
        x = torch.from_numpy(np.array([ob], dtype=np.float32)).permute(0, 3, 1, 2)
        cnn_out = self.cnn.forward(x=x)
        return self.ctrnn.step(ob=cnn_out.numpy())

    @classmethod
    def _get_sub_individual_size(cls, config, input_space, output_space):
        cnn_output_space = Cnn.get_output_shape(config=config.cnn_conf, input_space=input_space)

        cnn_size = Cnn.get_individual_size(config=config.cnn_conf, input_space=input_space,
                                           output_space=cnn_output_space)
        ctrnn_size = ContinuousTimeRNN.get_individual_size(config=config.ctrnn_conf, input_space=cnn_output_space,
                                                           output_space=output_space)
        return cnn_size, ctrnn_size, cnn_output_space

    @classmethod
    def get_individual_size(cls, config: CnnCtrnnCfg, input_space: Space, output_space: Space):
        a, b, _ = cls._get_sub_individual_size(config, input_space, output_space)
        return a + b

    @classmethod
    def set_masks_globally(cls, config: CnnCtrnnCfg, input_space: Space, output_space: Space):
        cnn_output_space = Cnn.get_output_shape(config=config.cnn_conf, input_space=input_space)
        ContinuousTimeRNN.set_masks_globally(config.ctrnn_conf, cnn_output_space, output_space)

    @classmethod
    def get_class_state(cls):
        return ContinuousTimeRNN.get_class_state()

    @classmethod
    def set_class_state(cls, v_mask, w_mask, t_mask):
        # https://github.com/pytorch/pytorch/issues/24398
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        logging.debug("Setting number of torch-threads to 1")
        # see https://github.com/pytorch/pytorch/issues/13757
        torch.set_num_threads(1)
        return ContinuousTimeRNN.set_class_state(v_mask, w_mask, t_mask)


class Cnn(nn.Module):
    _number_input_channels = 3

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConvolutionalNNCfg):
        super().__init__()
        assert len(individual) == self.get_individual_size(config, input_space, output_space)

        with torch.no_grad():
            self.conv1, self.conv2 = self._make_convs(config)
            self.maxpool1 = nn.MaxPool2d(kernel_size=config.maxp_size1, stride=config.maxp_stride1)
            self.maxpool2 = nn.MaxPool2d(kernel_size=config.maxp_size2, stride=config.maxp_stride2)

            index = 0
            size = self.conv1.weight.size().numel()
            weight = individual[index:index + size]
            self.conv1.weight = nn.Parameter(torch.from_numpy(np.array(weight)).view(self.conv1.weight.size()).float())
            index += size

            size = self.conv2.weight.size().numel()
            weight = individual[index:index + size]
            self.conv2.weight = nn.Parameter(torch.from_numpy(np.array(weight)).view(self.conv2.weight.size()).float())
            index += size
            assert index == len(individual), "Size of individual doesn't match number of weights needed"

        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = torch.sigmoid(self.conv2(x))
            x = self.maxpool2(x)
            return x.flatten()

    @classmethod
    def _make_convs(cls, config: ConvolutionalNNCfg):
        # This method exists because DRY and convs are needed before create the individuals
        conv1 = nn.Conv2d(cls._number_input_channels, config.conv_feat1,
                          kernel_size=config.conv_size1,
                          stride=config.conv_stride1)
        conv2 = nn.Conv2d(config.conv_feat1, config.conv_feat2,
                          kernel_size=config.conv_size2,
                          stride=config.conv_stride2)
        return conv1, conv2

    @classmethod
    def get_individual_size(cls, config: ConvolutionalNNCfg, input_space: Space, output_space: Space):
        assert output_space == cls.get_output_shape(config, input_space), "output space is not what it should be"
        conv1, conv2 = cls._make_convs(config)
        return conv1.weight.size().numel() + conv2.weight.size().numel()

    @classmethod
    def get_output_shape(cls, config: ConvolutionalNNCfg, input_space: Space) -> Space:
        h = input_space.shape[0]
        w = input_space.shape[1]
        d = input_space.shape[2]
        assert d == cls._number_input_channels, "Wrong input shape. CNN expects 2D image with 3 colors channels"
        h, w = cls._conv_output_shape(h_w=(h, w), kernel_size=config.conv_size1, stride=config.conv_stride1)
        h, w = cls._conv_output_shape(h_w=(h, w), kernel_size=config.maxp_size1, stride=config.maxp_stride1)
        h, w = cls._conv_output_shape(h_w=(h, w), kernel_size=config.conv_size2, stride=config.conv_stride2)
        h, w = cls._conv_output_shape(h_w=(h, w), kernel_size=config.maxp_size2, stride=config.maxp_stride2)
        return Box(-1, 1, (h, w, config.conv_feat2), np.float32)

    @classmethod
    def _conv_output_shape(cls, h_w: Union[tuple, int],
                           kernel_size: Union[tuple, int],
                           stride: Union[tuple, int],
                           pad: Union[tuple, int] = 0,
                           dilation=1):
        # source https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
        """
        Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """

        if type(h_w) is not tuple:
            h_w = (h_w, h_w)

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)

        if type(stride) is not tuple:
            stride = (stride, stride)

        if type(pad) is not tuple:
            pad = (pad, pad)

        h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
        w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

        return h, w
