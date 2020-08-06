from brains.ffnn import FeedForwardPyTorch, FeedForwardNumPy
from tools.configurations import FeedForwardCfg
import gym
import numpy as np
import time
import torch


class TestFeedForwardNN:

    def test_lstm_output(self, ffnn_config: FeedForwardCfg):

        input_size = 28
        output_size = 8

        number_of_inputs = 10000

        input_space = gym.spaces.Box(-1, 1, (input_size,))
        output_space = gym.spaces.Box(-1, 1, (output_size,))

        individual_size = FeedForwardPyTorch.get_individual_size(ffnn_config, input_space, output_space)

        # Make two copies to avoid possible errors due to PyTorch reusing data from the same memory address
        individual_pytorch = np.random.randn(individual_size).astype(np.float32)
        individual_numpy = np.copy(individual_pytorch)

        with torch.no_grad():
            ffnn_pytorch = FeedForwardPyTorch(input_space, output_space, individual_pytorch, ffnn_config)

        ffnn_numpy = FeedForwardNumPy(input_space, output_space, individual_numpy, ffnn_config)

        with torch.no_grad():
            # Hidden layers are in a stacked list, so unpack first
            hidden_layers = ffnn_config.hidden_layers[0]
            current_input = input_size

            for i, hidden_layer in enumerate(hidden_layers):
                weight_array_pytorch: np.ndarray = ffnn_pytorch.layers[i].weight.data.numpy()
                weight_array_numpy: np.ndarray = ffnn_numpy.weights[i]

                assert np.array_equal(weight_array_pytorch, weight_array_numpy)
                assert weight_array_pytorch.shape == (hidden_layer, current_input)

                if ffnn_config.use_bias:
                    bias_array_pytorch: np.ndarray = ffnn_pytorch.layers[i].bias.data.numpy()
                    bias_array_numpy: np.ndarray = ffnn_numpy.bias[i]

                    assert np.array_equal(bias_array_pytorch, bias_array_numpy)
                    assert bias_array_pytorch.shape == (hidden_layer,)

                current_input = hidden_layer

            assert weight_array_pytorch.shape == (current_input, output_size)

        inputs = np.random.randn(number_of_inputs, input_size).astype(np.float32)

        ffnn_pytorch_outputs = []
        ffnn_numpy_outputs = []

        ffnn_pytorch_times = []
        ffnn_numpy_times = []

        # Collect predictions of PyTorch and NumPy implementations and collect time data
        for i in inputs:
            with torch.no_grad():
                time_s = time.time()
                ffnn_pytorch_output = ffnn_pytorch.step(i)
                ffnn_pytorch_times.append(time.time() - time_s)
                ffnn_pytorch_outputs.append(ffnn_pytorch_output)

            time_s = time.time()
            ffnn_numpy_output = ffnn_numpy.step(i)
            ffnn_numpy_times.append(time.time() - time_s)
            ffnn_numpy_outputs.append(ffnn_numpy_output)

        ffnn_pytorch_outputs = np.array(ffnn_pytorch_outputs)
        ffnn_numpy_outputs = np.array(ffnn_numpy_outputs)

        ffnn_pytorch_times = np.array(ffnn_pytorch_times)
        ffnn_numpy_times = np.array(ffnn_numpy_times)

        assert len(ffnn_pytorch_outputs) == len(ffnn_numpy_outputs)
        assert ffnn_pytorch_outputs.size == ffnn_numpy_outputs.size

        print("\nPyTorch Mean Prediction Time {}s | NumPy Mean Prediction Time {}s"
              .format(np.mean(ffnn_pytorch_times), np.mean(ffnn_numpy_times)))

        print("PyTorch Stddev Prediction Time {}s | NumPy Stddev Prediction Time {}s"
              .format(np.std(ffnn_pytorch_times), np.std(ffnn_numpy_times)))

        print("PyTorch Max Prediction Time {}s | NumPy Max Prediction Time {}s"
              .format(np.max(ffnn_pytorch_times), np.max(ffnn_numpy_times)))

        print("PyTorch Min Prediction Time {}s | NumPy Min Prediction Time {}s"
              .format(np.min(ffnn_pytorch_times), np.min(ffnn_numpy_times)))

        # Use percentage instead of np.allclose() because some results exceed the rtol value, but it is a low percentage
        close_percentage = np.count_nonzero(
            np.isclose(ffnn_pytorch_outputs, ffnn_numpy_outputs)) / ffnn_pytorch_outputs.size

        assert close_percentage > 0.98

        print(
            "Equal predictions between PyTorch and NumPy",
            "Implementation of FFNN: {}% of {} predictions".format(close_percentage*100, number_of_inputs))
