import pytest
import os
from tools.helper import config_from_file
from tools.configurations import ExperimentCfg, ContinuousTimeRNNCfg
from brains.layered_nn import LayeredNNCfg
from gym.spaces import Box


@pytest.fixture
def box2d():
    return Box(-1, 1, shape=[2])


@pytest.fixture
def config() -> ExperimentCfg:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    config_location = os.path.join(current_directory, "basic_test_config.json")
    global_config = config_from_file(config_location)
    return global_config


@pytest.fixture
def ctrnn_config(config: ExperimentCfg) -> ContinuousTimeRNNCfg:
    return config.brain


@pytest.fixture
def lnn_config() -> LayeredNNCfg:
    return LayeredNNCfg(type="LNN", number_neurons_layer1=2, number_neurons_layer2=2, cppn_hidden_size1=2,
                        cppn_hidden_size2=2, use_biases=True, indirect_encoding=False)
