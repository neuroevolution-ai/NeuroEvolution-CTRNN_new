import numpy as np
from gym import spaces

from brains.ffnn import FeedForwardNumPy, FeedForwardPyTorch


class TestIBrain:
    low = -5.0
    high = 5.0
    one_dimensional_shape = (20,)
    multi_dimensional_shape = (30, 15)
    input_n = 50
    output_n = 10

    # @pytest.mark.skip(reason="Not yet implemented")
    def test_parse_box_space(self, ffnn_config):
        input_space = spaces.Box(low=self.low, high=self.high, shape=self.one_dimensional_shape)
        output_space = spaces.Box(low=self.low, high=self.high, shape=self.multi_dimensional_shape)

        # First test the input space

        # Not used but needed to create a Brain
        dummy_individual = np.zeros((FeedForwardNumPy.get_individual_size(
            ffnn_config, input_space=input_space, output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space, output_space=output_space, individual=dummy_individual,
                                   config=ffnn_config)

        sample_input = input_space.sample()
        parsed_input = brain.parse_brain_input_output(sample_input, is_brain_input=True)

        # When input_space == spaces.Box the input is not transformed
        assert np.array_equal(sample_input, parsed_input)

        # Environments with RGB observations usually use spaces.Box as their observation space, which means it is
        # the input space for the brains
        # Assume now we have RGB input
        sample_input = input_space.sample().reshape((1, 1, -1))
        parsed_input = brain.parse_brain_input_output(sample_input, is_brain_input=True)

        # When input_space == spaces.Box the input is not transformed
        assert np.array_equal(sample_input, parsed_input)
        assert isinstance(parsed_input, np.ndarray) and len(parsed_input.shape) == 3

        # Now test the output space

        # Reference shape, this is what the output of the brain should look like
        sample_output_shape = output_space.sample().shape

        brain_output = brain.step(input_space.sample())

        assert brain_output.shape == sample_output_shape

    def test_parse_discrete_space(self, ffnn_config):
        input_space = spaces.Discrete(n=self.input_n)
        output_space = spaces.Discrete(n=self.output_n)

        dummy_individual = np.zeros((FeedForwardNumPy.get_individual_size(
            ffnn_config, input_space=input_space, output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space, output_space=output_space, individual=dummy_individual,
                                   config=ffnn_config)

        # Test input space

        sample_input = input_space.sample()
        parsed_input = brain.parse_brain_input_output(sample_input, is_brain_input=True)

        # Parsed Input should be one-hot encoded
        assert len(parsed_input.shape) == 1
        assert len(parsed_input) == input_space.n
        assert all(x == 0 for x in parsed_input[:sample_input])
        assert parsed_input[sample_input] == 1
        assert all(x == 0 for x in parsed_input[sample_input + 1:])

        # Test output space

        brain_output = brain.step(input_space.sample())

        # This should be an Integer in [0, output_space.n - 1]
        assert isinstance(brain_output, np.integer)
        assert 0 <= brain_output < output_space.n

    def test_parse_tuple_space(self, ffnn_config):
        # Prepare some spaces to test the Tuple space
        one_dimensional_box = spaces.Box(low=self.low, high=self.high, shape=self.one_dimensional_shape)
        multi_dimensional_box = spaces.Box(low=self.low, high=self.high, shape=self.multi_dimensional_shape)
        first_discrete = spaces.Discrete(n=self.input_n)
        second_discrete = spaces.Discrete(n=self.output_n)
        nested_tuple = spaces.Tuple([one_dimensional_box, multi_dimensional_box])

        input_space = spaces.Tuple([nested_tuple, one_dimensional_box, first_discrete])
        output_space = spaces.Tuple([nested_tuple, multi_dimensional_box, second_discrete])

        dummy_individual = np.zeros((FeedForwardNumPy.get_individual_size(
            ffnn_config, input_space=input_space, output_space=output_space)))

        brain = FeedForwardPyTorch(input_space=input_space, output_space=output_space, individual=dummy_individual,
                                   config=ffnn_config)

        # Test input space

        sample_input = input_space.sample()
        parsed_input = brain.parse_brain_input_output(sample_input, is_brain_input=True)

        assert len(parsed_input) == 3
        assert len(parsed_input[0]) == 2
        assert parsed_input[0][0].shape == one_dimensional_box.shape
        assert parsed_input[0][1].shape == multi_dimensional_box.shape
        assert parsed_input[1].shape == one_dimensional_box.shape
        assert len(parsed_input[2].shape) == 1  # One-Hot encoded, rest of one-hot encoding is tested in other test case
        assert len(parsed_input[2]) == first_discrete.n

        # Test output space

        brain_output = brain.step(input_space.sample())

        # See definition of output_space
        assert len(brain_output) == 3
        assert len(brain_output[0]) == 2
        assert brain_output[0][0].shape == one_dimensional_box.shape
        assert brain_output[0][1].shape == multi_dimensional_box.shape
        assert brain_output[1].shape == multi_dimensional_box.shape
        assert isinstance(brain_output[2], np.integer)
        assert 0 <= brain_output[2] < second_discrete.n
