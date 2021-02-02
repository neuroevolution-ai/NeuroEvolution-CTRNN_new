import pytest
import os
from tools.config_reader import ConfigReader
from tools.configurations import ExperimentCfg, ContinuousTimeRNNCfg, LSTMCfg, FeedForwardCfg, ConcatenatedBrainLSTMCfg, \
    ILayerBasedBrainCfg
from tools.configurations import ExperimentCfg, ContinuousTimeRNNCfg, LSTMCfg, FeedForwardCfg, ConcatenatedBrainLSTMCfg, \
    OptimizerMuLambdaCfg
from gym.spaces import Box


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
    return OptimizerMuLambdaCfg(type='MU_ES', initial_gene_range=2, mu=2, lambda_=3, mutpb=0.5)


@pytest.fixture
def ffnn_config() -> FeedForwardCfg:
    return FeedForwardCfg(type="FeedForward_PyTorch", normalize_input=False, normalize_input_target=1, use_bias=True,
                          hidden_layers=[[8, 16]], non_linearity="relu", indirect_encoding=False,
                          cppn_hidden_layers=[[2, 4]])


@pytest.fixture
def lstm_config() -> LSTMCfg:
    return LSTMCfg(type="LSTM_PyTorch", normalize_input=False, normalize_input_target=1, lstm_num_layers=3,
                   use_bias=True)


@pytest.fixture
def layer_config() -> ILayerBasedBrainCfg:
    return ILayerBasedBrainCfg(type=None, normalize_input=False, normalize_input_target=1, use_bias=True,
                               hidden_layer_structure=[2], diagonal_hidden_to_hidden=False,
                               optimize_initial_hidden_values=True, )


@pytest.fixture
def concat_lstm_config() -> ConcatenatedBrainLSTMCfg:
    return ConcatenatedBrainLSTMCfg(type="ConcatenatedBrain_LSTM", normalize_input=False, normalize_input_target=1,
                                    use_bias=True,
                                    feed_forward_front=FeedForwardCfg(type="FeedForward_PyTorch", normalize_input=False,
                                                                      normalize_input_target=1, use_bias=True,
                                                                      hidden_layers=[8, 16], non_linearity="relu",
                                                                      indirect_encoding=False,
                                                                      cppn_hidden_layers=[2, 4]),
                                    lstm=LSTMCfg(type="LSTM_PyTorch", normalize_input=False, normalize_input_target=1,
                                                 lstm_num_layers=3, use_bias=True),
                                    feed_forward_back=FeedForwardCfg(type="FeedForward_PyTorch", normalize_input=False,
                                                                     normalize_input_target=1, use_bias=True,
                                                                     hidden_layers=[32, 64], non_linearity="relu",
                                                                     indirect_encoding=False,
                                                                     cppn_hidden_layers=[2, 4]))
