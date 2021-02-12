import numpy as np
from gym import spaces

from brains.ffnn import FeedForwardNumPy, FeedForwardPyTorch


class TestIBrain:
    low = -5.0
    high = 5.0
    one_dimensional_shape = (20, )
    rgb_shape = (3, 32, 32)
    multi_dimensional_shape = (30, 15)
    input_n = 50
    output_n = 10

    def test_parse_box_space(self, ffnn_config):
        input_space = spaces.Box(low=self.low,
                                 high=self.high,
                                 shape=self.one_dimensional_shape)
        output_space = spaces.Box(low=self.low,
                                  high=self.high,
                                  shape=self.multi_dimensional_shape)

        # First test the input space
        # Not used but needed to create a Brain
        dummy_individual = np.zeros(
            (FeedForwardNumPy.get_individual_size(ffnn_config,
                                                  input_space=input_space,
                                                  output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space,
                                   output_space=output_space,
                                   individual=dummy_individual,
                                   config=ffnn_config)

        sample_input = np.ones(input_space.shape)
        assert input_space.contains(sample_input)
        parsed_input = brain.parse_brain_input(sample_input)

        # When input_space == spaces.Box the input is not transformed
        assert np.array_equal(sample_input, parsed_input)

        # Now test the output space
        brain_output = brain.step(sample_input)

        assert brain_output.shape == output_space.shape

    def test_parse_rgb_input(self, ffnn_config):
        # Environments with RGB observations usually use spaces.Box as their observation space, which means it is
        # the input space for the brains
        input_space = spaces.Box(low=self.low,
                                 high=self.high,
                                 shape=self.rgb_shape)
        output_space = spaces.Box(low=self.low,
                                  high=self.high,
                                  shape=self.one_dimensional_shape)

        # First test the input space
        # Not used but needed to create a Brain
        dummy_individual = np.zeros(
            (FeedForwardNumPy.get_individual_size(ffnn_config,
                                                  input_space=input_space,
                                                  output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space,
                                   output_space=output_space,
                                   individual=dummy_individual,
                                   config=ffnn_config)

        sample_input = np.ones(input_space.shape)
        assert input_space.contains(sample_input)
        # Assume now we have RGB input space, in our flatten function this should return a multidimensional NumPy array
        parsed_input = brain.parse_brain_input(sample_input)

        # When input_space == spaces.Box the input is not transformed
        assert np.array_equal(sample_input, parsed_input)
        assert (isinstance(parsed_input, np.ndarray)
                and len(parsed_input.shape) == 3
                and parsed_input.shape == sample_input.shape)

    def test_parse_discrete_space(self, ffnn_config):
        input_space = spaces.Discrete(n=self.input_n)
        output_space = spaces.Discrete(n=self.output_n)

        dummy_individual = np.zeros(
            (FeedForwardNumPy.get_individual_size(ffnn_config,
                                                  input_space=input_space,
                                                  output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space,
                                   output_space=output_space,
                                   individual=dummy_individual,
                                   config=ffnn_config)

        # Test input space
        sample_input = 1
        assert input_space.contains(sample_input)
        parsed_input = brain.parse_brain_input(sample_input)

        # Parsed Input should be one-hot encoded
        assert len(parsed_input.shape) == 1
        assert len(parsed_input) == input_space.n
        assert all(x == 0 for x in parsed_input[:sample_input])
        assert parsed_input[sample_input] == 1
        assert all(x == 0 for x in parsed_input[sample_input + 1:])

        # Test output space
        brain_output = brain.step(sample_input)

        # This should be an Integer in [0, output_space.n - 1]
        assert isinstance(brain_output, np.integer)
        assert 0 <= brain_output < output_space.n

    def test_parse_tuple_space(self, ffnn_config):
        # Prepare some spaces to test the Tuple space
        one_dimensional_box = spaces.Box(low=self.low,
                                         high=self.high,
                                         shape=self.one_dimensional_shape)
        multi_dimensional_box = spaces.Box(low=self.low,
                                           high=self.high,
                                           shape=self.multi_dimensional_shape)
        first_discrete = spaces.Discrete(n=self.input_n)
        second_discrete = spaces.Discrete(n=self.output_n)

        input_space = spaces.Tuple([
            one_dimensional_box, first_discrete, multi_dimensional_box,
            second_discrete
        ])
        output_space = spaces.Tuple([
            second_discrete, multi_dimensional_box, one_dimensional_box,
            first_discrete
        ])

        dummy_individual = np.zeros(
            (FeedForwardNumPy.get_individual_size(ffnn_config,
                                                  input_space=input_space,
                                                  output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space,
                                   output_space=output_space,
                                   individual=dummy_individual,
                                   config=ffnn_config)

        # Test input space
        sample_input = (np.ones(self.one_dimensional_shape), 1,
                        np.ones(self.multi_dimensional_shape) * 2, 2)
        assert input_space.contains(sample_input)
        parsed_input = brain.parse_brain_input(sample_input)
        assert len(parsed_input.shape) == 1
        assert spaces.flatdim(input_space) == len(parsed_input)

        # Test output space
        brain_output = brain.step(sample_input)

        # 4 nested spaces inside the input space
        assert len(brain_output) == 4

        # First nested space is second_discrete
        nested_output = brain_output[0]
        assert isinstance(nested_output, np.integer)

        # Second nested space is multi_dimensional_box
        nested_output = brain_output[1]
        assert len(nested_output.shape) == len(self.multi_dimensional_shape)
        assert all([
            x == y
            for (x,
                 y) in zip(nested_output.shape, self.multi_dimensional_shape)
        ])

        # Third nested space is one_dimensional_box
        nested_output = brain_output[2]
        assert len(nested_output.shape) == len(self.one_dimensional_shape)
        assert all([
            x == y
            for (x, y) in zip(nested_output.shape, self.one_dimensional_shape)
        ])

        # Fourth nested space is first_discrete
        nested_output = brain_output[3]
        assert isinstance(nested_output, np.integer)
