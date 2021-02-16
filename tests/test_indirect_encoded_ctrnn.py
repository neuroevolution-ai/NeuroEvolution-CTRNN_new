import gym
import numpy as np

from brains.indirect_encoded_ctrnn import IndirectEncodedCtrnn
from tools.configurations import FeedForwardCfg, ContinuousTimeRNNCfg, IndirectEncodedCtrnnCfg

class TestIndirectEncodedCTRNN:

    def test_hyperbrain(self, ffnn_config: FeedForwardCfg):

        input_size = 20
        output_size = 10

        input_space = gym.spaces.Box(-1, 1, (input_size,))
        output_space = gym.spaces.Box(-1, 1, (output_size,))

        hyperbrain_config = IndirectEncodedCtrnnCfg(type="Indirect_CTRNN",
                                                    ffnn_conf=FeedForwardCfg(type="FeedForward_Numpy", use_bias=True,
                                                                             hidden_layers=[[16, 32, 16, 8]],
                                                                             neuron_activation="relu",
                                                                             neuron_activation_output="linear"),
                                                    ctrnn_conf=ContinuousTimeRNNCfg(type="CTRNN",
                                                                                    optimize_initial_neuron_state=False,
                                                                                    delta_t=0.05,
                                                                                    optimize_state_boundaries="fixed",
                                                                                    number_neurons=200,
                                                                                    neuron_activation="tanh",
                                                                                    set_principle_diagonal_elements_of_W_negative=True))

        IndirectEncodedCtrnn.generate_and_set_class_state(config=hyperbrain_config, input_space=input_space, output_space=output_space)

        individual_size = IndirectEncodedCtrnn.get_individual_size(hyperbrain_config, input_space, output_space)
        assert individual_size == 1329

        individual = np.random.rand(individual_size).astype(np.float32)

        hyperbrain = IndirectEncodedCtrnn(input_space, output_space, individual, hyperbrain_config)


