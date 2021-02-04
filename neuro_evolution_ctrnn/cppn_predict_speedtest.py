from brains.ffnn import FeedForwardPyTorch, FeedForwardNumPy
from tools.configurations import FeedForwardCfg
import numpy as np
import gym
import time

input_size = 6
output_size = 1

number_of_inputs = 3000000

input_space = gym.spaces.Box(-1, 1, (input_size,))
output_space = gym.spaces.Box(-1, 1, (output_size,))

ffnn_config = FeedForwardCfg(type="FeedForward_Numpy", normalize_input=False, normalize_input_target=1, use_bias=True,
                             hidden_layers=[[16, 32, 16, 8]], neuron_activation="relu",
                             neuron_activation_output="linear")

individual_size = FeedForwardPyTorch.get_individual_size(ffnn_config, input_space, output_space)
individual = np.random.randn(individual_size).astype(np.float32)

ffnn_numpy = FeedForwardNumPy(input_space, output_space, individual, ffnn_config)

#inputs = np.random.randn(number_of_inputs, input_size).astype(np.float32)
inputs = np.random.rand(number_of_inputs, input_size, 1).astype(np.float32)

t1 = time.time()

ffnn_numpy_output = ffnn_numpy.step(inputs)

t2 = time.time()

print((t2-t1))

print("finished")