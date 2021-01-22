import numpy as np
import pytest
from gym import spaces

from brains.ffnn import FeedForwardNumPy, FeedForwardPyTorch


class TestIBrain:

    # @pytest.mark.skip(reason="Not yet implemented")
    def test_parse_box_space(self, ffnn_config):
        input_space = spaces.Box(-5.0, +5.0, (20,))
        output_space = spaces.Box(-5.0, +5.0, (30, 15))

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

        assert brain_output.shape == sample_output_shape.shape
