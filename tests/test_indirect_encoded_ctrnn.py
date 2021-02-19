import gym
import numpy as np
from gym.spaces import flatdim
import time

from brains.indirect_encoded_ctrnn import IndirectEncodedCtrnn
from tools.configurations import FeedForwardCfg, ContinuousTimeRNNCfg, IndirectEncodedCtrnnCfg

class TestIndirectEncodedCTRNN:

    def test_hyperbrain(self):

        env = gym.make('procgen:procgen-heist-v0')

        input_space = env.observation_space
        output_space = env.action_space

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
                                                                                    set_principle_diagonal_elements_of_W_negative=False))

        IndirectEncodedCtrnn.generate_and_set_class_state(config=hyperbrain_config, input_space=input_space,
                                                          output_space=output_space)

        individual_size = IndirectEncodedCtrnn.get_individual_size(hyperbrain_config, input_space, output_space)
        assert individual_size == 1329

        individual = np.random.rand(individual_size).astype(np.float32)

        t_start = time.time()
        hyperbrain = IndirectEncodedCtrnn(input_space, output_space, individual, hyperbrain_config)
        print("Initializing indirect encoded CTRNN took {}s".format(time.time() - t_start))

        input_image_width = input_space.shape[0]
        input_image_height = input_space.shape[1]
        input_size = input_image_width*input_image_height*3
        number_neurons = hyperbrain.config.ctrnn_conf.number_neurons
        output_size = flatdim(output_space)

        assert hyperbrain.input_positions.ndim == 2
        assert hyperbrain.input_positions.shape[0] == input_size
        assert hyperbrain.input_positions.shape[1] == 3

        assert hyperbrain.neuron_positions.ndim == 2
        assert hyperbrain.neuron_positions.shape[0] == number_neurons
        assert hyperbrain.neuron_positions.shape[1] == 3

        assert hyperbrain.output_positions.ndim == 2
        assert hyperbrain.output_positions.shape[0] == output_size
        assert hyperbrain.output_positions.shape[1] == 3

        # Decode V using cppn and compare
        V_test = np.zeros((number_neurons, input_size))
        for i in range(len(hyperbrain.neuron_positions)):
            for j in range(len(hyperbrain.input_positions)):

                cppn_input = np.concatenate((hyperbrain.input_positions[j], hyperbrain.neuron_positions[i]))
                assert cppn_input.ndim == 1
                assert cppn_input.size == 6

                V_test[i, j] = hyperbrain.cppn.step(cppn_input)

        assert np.array_equal(V_test, hyperbrain.ctrnn.V.toarray())

        # Decode W using cppn and compare
        W_test = np.zeros((number_neurons, number_neurons))
        for i in range(len(hyperbrain.neuron_positions)):
            for j in range(len(hyperbrain.neuron_positions)):

                cppn_input = np.concatenate((hyperbrain.neuron_positions[j], hyperbrain.neuron_positions[i]))
                assert cppn_input.ndim == 1
                assert cppn_input.size == 6

                W_test[i, j] = hyperbrain.cppn.step(cppn_input)

        assert np.array_equal(W_test, hyperbrain.ctrnn.W.toarray())

        # Decode T using cppn and compare
        T_test = np.zeros((output_size, number_neurons))
        for i in range(len(hyperbrain.output_positions)):
            for j in range(len(hyperbrain.neuron_positions)):

                cppn_input = np.concatenate((hyperbrain.neuron_positions[j], hyperbrain.output_positions[i]))
                assert cppn_input.ndim == 1
                assert cppn_input.size == 6

                T_test[i, j] = hyperbrain.cppn.step(cppn_input)

        assert np.array_equal(T_test, hyperbrain.ctrnn.T.toarray())

        obs = env.reset()

        while True:
            action = hyperbrain.step(obs)
            obs, rew, done, info = env.step(action)

            if done:
                break
