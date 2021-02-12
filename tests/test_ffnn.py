import time

import gym
import numpy as np
import torch

from brains.ffnn import FeedForwardPyTorch, FeedForwardNumPy
from tools.configurations import FeedForwardCfg


class TestFeedForwardNN:

    def test_ffnn_output(self, ffnn_config: FeedForwardCfg):

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
                weight_array_numpy: np.ndarray = ffnn_numpy.weights_hidden_layers[i]

                assert np.array_equal(weight_array_pytorch, weight_array_numpy)
                assert weight_array_pytorch.shape == (hidden_layer, current_input)

                if ffnn_config.use_bias:
                    bias_array_pytorch: np.ndarray = ffnn_pytorch.layers[i].bias.data.numpy()
                    bias_array_numpy: np.ndarray = ffnn_numpy.biases_hidden_layers[i]

                    assert np.array_equal(bias_array_pytorch, bias_array_numpy.flatten())
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
        ffnn_numpy_predict_outputs = ffnn_numpy.predict(inputs.reshape(number_of_inputs, input_size, 1))

        ffnn_pytorch_times = np.array(ffnn_pytorch_times)
        ffnn_numpy_times = np.array(ffnn_numpy_times)

        assert len(ffnn_pytorch_outputs) == len(ffnn_numpy_outputs)
        assert len(ffnn_pytorch_outputs) == len(ffnn_numpy_predict_outputs)
        assert ffnn_pytorch_outputs.size == ffnn_numpy_outputs.size
        assert ffnn_pytorch_outputs.size == ffnn_numpy_predict_outputs.size

        # Note: {:f} is used to suppress scientific notation (i.e. displaying 0.00001 instead of 1e-5)
        print("\nPyTorch Mean Prediction Time {:f} ms | NumPy Mean Prediction Time {:f} ms"
              .format(np.mean(ffnn_pytorch_times)*1000, np.mean(ffnn_numpy_times)*1000))

        print("PyTorch Stddev Prediction Time {:f} ms | NumPy Stddev Prediction Time {:f} ms"
              .format(np.std(ffnn_pytorch_times)*1000, np.std(ffnn_numpy_times)*1000))

        print("PyTorch Max Prediction Time {:f} ms | NumPy Max Prediction Time {:f} ms"
              .format(np.max(ffnn_pytorch_times)*1000, np.max(ffnn_numpy_times)*1000))

        print("PyTorch Min Prediction Time {:f} ms | NumPy Min Prediction Time {:f} ms"
              .format(np.min(ffnn_pytorch_times)*1000, np.min(ffnn_numpy_times)*1000))

        # Use percentage instead of np.allclose() because some results exceed the rtol value, but it is a low percentage
        close_percentage = np.count_nonzero(
            np.isclose(ffnn_pytorch_outputs, ffnn_numpy_outputs)) / ffnn_pytorch_outputs.size

        assert close_percentage > 0.98

        print(
            "Equal predictions between PyTorch and NumPy",
            "Implementation of FFNN: {}% of {} predictions".format(close_percentage*100, number_of_inputs))

        assert ffnn_numpy_outputs.size == np.count_nonzero(
            np.isclose(ffnn_numpy_predict_outputs.reshape(number_of_inputs, output_size),
                       ffnn_numpy_outputs))

    def test_ffnn_predict_speedtest(self):

        def predict_with_maximum_speed(u, A, B, C, D, E, a, b, c, d, e):
            x = np.matmul(A, u)
            x = np.add(x, a)
            x = np.maximum(0, x)

            x = np.matmul(B, x)
            x = np.add(x, b)
            x = np.maximum(0, x)

            x = np.matmul(C, x)
            x = np.add(x, c)
            x = np.maximum(0, x)

            x = np.matmul(D, x)
            x = np.add(x, d)
            x = np.maximum(0, x)

            x = np.matmul(E, x)
            x = np.add(x, e)

            return x

        input_size = 6
        output_size = 1

        number_of_inputs = 3000000

        input_space = gym.spaces.Box(-1, 1, (input_size,))
        output_space = gym.spaces.Box(-1, 1, (output_size,))

        ffnn_config = FeedForwardCfg(type="FeedForward_Numpy",
                                     use_bias=True,
                                     hidden_layers=[[16, 32, 16, 8]], neuron_activation="relu",
                                     neuron_activation_output="linear")

        individual_size = FeedForwardPyTorch.get_individual_size(ffnn_config, input_space, output_space)
        individual = np.random.rand(individual_size).astype(np.float32)

        ffnn_numpy = FeedForwardNumPy(input_space, output_space, individual, ffnn_config)

        # Weight Matrizes
        A = ffnn_numpy.weights_hidden_layers[0].copy()
        B = ffnn_numpy.weights_hidden_layers[1].copy()
        C = ffnn_numpy.weights_hidden_layers[2].copy()
        D = ffnn_numpy.weights_hidden_layers[3].copy()
        E = ffnn_numpy.weights_output_layer.copy()

        # Biases
        a = ffnn_numpy.biases_hidden_layers[0].copy()
        b = ffnn_numpy.biases_hidden_layers[1].copy()
        c = ffnn_numpy.biases_hidden_layers[2].copy()
        d = ffnn_numpy.biases_hidden_layers[3].copy()
        e = ffnn_numpy.biases_output_layer.copy()

        inputs = np.random.rand(number_of_inputs, input_size, 1).astype(np.float32)

        t_start = time.time()
        output_ffnn_class = ffnn_numpy.predict(inputs)
        time_ffnn_class = time.time() - t_start
        print(time_ffnn_class)

        t_start = time.time()
        output_maximum_speed = predict_with_maximum_speed(inputs, A, B, C, D, E, a, b, c, d, e)
        time_maximum_speed = time.time() - t_start
        print(time_maximum_speed)

        relative_speed_tolerance = time_ffnn_class / time_maximum_speed - 1

        print(relative_speed_tolerance)

        print( np.count_nonzero( np.isclose( output_ffnn_class, output_maximum_speed)))
        assert number_of_inputs == np.count_nonzero( np.isclose( output_ffnn_class, output_maximum_speed))
