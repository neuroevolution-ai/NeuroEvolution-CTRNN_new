from brains.lstm_pytorch import LSTMPyTorch
from brains.lstm_numpy import LSTMNumPy
from tools.configurations import ExperimentCfg, LSTMCfg

import numpy as np
import gym
import torch


class TestLSTM:

    def test_lstm_output(self, lstm_config: LSTMCfg):

        input_size = 28
        output_size = 8

        number_of_inputs = 10

        input_space = gym.spaces.Box(-1, 1, (input_size,))
        output_space = gym.spaces.Box(-1, 1, (output_size,))

        individual_size = LSTMPyTorch.get_individual_size(lstm_config, input_space, output_space)

        # Make two copies to avoid possible errors due to PyTorch reusing data from the same memory address
        individual_pytorch = np.random.randn(individual_size).astype(np.float32)
        individual_numpy = np.copy(individual_pytorch)

        lstm_pytorch = LSTMPyTorch(input_space, output_space, individual_pytorch, lstm_config)

        lstm_numpy = LSTMNumPy(input_space, output_space, individual_numpy, lstm_config)

        assert np.array_equal(lstm_pytorch.lstm.weight_ih_l0.detach().numpy(), lstm_numpy.weight_ih_l0)
        assert np.array_equal(lstm_pytorch.lstm.weight_hh_l0.detach().numpy(), lstm_numpy.weight_hh_l0)

        assert np.array_equal(lstm_pytorch.lstm.bias_hh_l0.detach().numpy(), lstm_numpy.bias_hh_l0)
        assert np.array_equal(lstm_pytorch.lstm.bias_hh_l0.detach().numpy(), lstm_numpy.bias_hh_l0)

        # Also initialize the values for the hidden and cell states the same
        hidden_pytorch = np.random.randn(*lstm_pytorch.hidden[0].size()).astype(np.float32)
        cell_pytorch = np.random.randn(*lstm_pytorch.hidden[1].size()).astype(np.float32)

        hidden_numpy = np.copy(hidden_pytorch).reshape(1, -1)
        cell_numpy = np.copy(cell_pytorch).reshape(1, -1)

        lstm_pytorch.hidden = (torch.from_numpy(hidden_pytorch), torch.from_numpy(cell_pytorch))

        lstm_numpy.hidden = np.copy(hidden_numpy)
        lstm_numpy.cell_state = np.copy(cell_numpy)

        inputs = np.random.randn(number_of_inputs, 28).astype(np.float32)

        lstm_pytorch_outputs = []
        lstm_numpy_outputs = []

        for i in inputs:
            lstm_pytorch_output = lstm_pytorch.step(i)
            lstm_pytorch_outputs.append(lstm_pytorch_output)

            lstm_numpy_output = lstm_numpy.step(i)
            lstm_numpy_outputs.append(lstm_numpy_output)

        lstm_pytorch_outputs = np.array(lstm_pytorch_outputs)
        lstm_numpy_outputs = np.array(lstm_numpy_outputs)

        print("PyTorch: ", lstm_pytorch_outputs)
        print("------------------------------------")
        print("NumPy: ", lstm_numpy_outputs)

        assert np.allclose(lstm_pytorch_outputs, lstm_numpy_outputs)
