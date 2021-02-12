import os

import pytest
from gym.spaces import Box

from tools.config_reader import ConfigReader
from tools.configurations import (ExperimentCfg, ContinuousTimeRNNCfg,
                                  FeedForwardCfg, ConcatenatedBrainLSTMCfg,
                                  OptimizerMuLambdaCfg, ILayerBasedBrainCfg,
                                  LstmTorchCfg, LstmLayeredCfg)


@pytest.fixture
def box2d():
    return Box(-1, 1, shape=[2])


@pytest.fixture
def config() -> ExperimentCfg:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    config_location = os.path.join(current_directory, "basic_test_config.json")
    global_config = ConfigReader.config_from_file(config_location)
    return global_config


@pytest.fixture
def ctrnn_config(config: ExperimentCfg) -> ContinuousTimeRNNCfg:
    return config.brain


@pytest.fixture
def mu_lambda_es_config() -> OptimizerMuLambdaCfg:
    return OptimizerMuLambdaCfg(type='MU_ES',
                                initial_gene_range=2,
                                mu=2,
                                lambda_=3,
                                mutpb=0.5)


@pytest.fixture
def ffnn_config() -> FeedForwardCfg:
    return FeedForwardCfg(type="FeedForward_PyTorch",
                          use_bias=True,
                          hidden_layers=[[8, 16]],
                          neuron_activation="relu")


@pytest.fixture
def lstm_torch_config() -> LstmTorchCfg:
    return LstmTorchCfg(type="LSTM_PyTorch",
                        num_layers=3,
                        hidden_size=8,
                        use_bias=False)


@pytest.fixture
def lstm_layered_config() -> LstmLayeredCfg:
    return LstmLayeredCfg(type="LSTMLayered",
                          hidden_layer_structure=[8, 8, 8],
                          diagonal_hidden_to_hidden=False,
                          use_bias=False,
                          optimize_initial_neuron_state=False)


@pytest.fixture
def layer_config() -> ILayerBasedBrainCfg:
    return ILayerBasedBrainCfg(type="",
                               use_bias=True,
                               hidden_layer_structure=[2],
                               diagonal_hidden_to_hidden=False,
                               optimize_initial_neuron_state=True)


@pytest.fixture
def concat_lstm_config() -> ConcatenatedBrainLSTMCfg:
    return ConcatenatedBrainLSTMCfg(
        type="ConcatenatedBrain_LSTM",
        use_bias=True,
        feed_forward_front=FeedForwardCfg(type="FeedForward_PyTorch",
                                          use_bias=True,
                                          hidden_layers=[8, 16],
                                          neuron_activation="relu"),
        lstm=LstmLayeredCfg(type="LSTM_PyTorch",
                            hidden_layer_structure=[8, 8, 8],
                            optimize_initial_neuron_state=False,
                            use_bias=True),
        feed_forward_back=FeedForwardCfg(type="FeedForward_PyTorch",
                                         use_bias=True,
                                         hidden_layers=[32, 64],
                                         neuron_activation="relu"))
