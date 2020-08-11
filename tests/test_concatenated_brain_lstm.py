from tools.configurations import ConcatenatedBrainLSTMCfg
from brains.concatenated_brains import ConcatenatedLSTM
from brains.lstm import LSTMNumPy
from brains.ffnn import FeedForwardNumPy

import numpy as np
from gym.spaces import Box


class TestConcatenatedLSTM:

    def test_concatenated_lstm_output(self, concat_lstm_config: ConcatenatedBrainLSTMCfg):
        input_size = 28
        output_size = 8

        input_space = Box(-1, 1, (input_size,))
        output_space = Box(-1, 1, (output_size,))

        # Create random individual
        individual_size = ConcatenatedLSTM.get_individual_size(concat_lstm_config, input_space, output_space)
        individual = np.random.randn(individual_size).astype(np.float32)

        concatenated_lstm = ConcatenatedLSTM(input_space, output_space, individual, concat_lstm_config)

        # Basic assertion to test if the architecture of the concatenated brain matches the chosen configuration
        assert (concatenated_lstm.feed_forward_front.hidden_layers ==
                concat_lstm_config.feed_forward_front.hidden_layers)

        assert concatenated_lstm.lstm.input_space.shape[0] == concat_lstm_config.feed_forward_front.hidden_layers[-1]

        assert (concatenated_lstm.feed_forward_back.hidden_layers ==
                concat_lstm_config.feed_forward_back.hidden_layers)

        # To test the concatenated brain, construct the individual parts alone and later compare the results
        # First construct the leading Feed Forward part
        ff_front_cfg = concat_lstm_config.feed_forward_front
        ff_front_output_space = Box(-1, 1, (ff_front_cfg.hidden_layers[-1],))
        ff_front_individual_size = FeedForwardNumPy.get_individual_size(ff_front_cfg, input_space, ff_front_output_space)

        current_index = 0
        ff_front_individual = individual[current_index:current_index + ff_front_individual_size]
        current_index += ff_front_individual_size

        feed_forward_front = FeedForwardNumPy(input_space, ff_front_output_space, ff_front_individual, ff_front_cfg)

        # Create input space for Feed Forward part at the back here because it is the output space for the LSTM
        ff_back_cfg = concat_lstm_config.feed_forward_back
        ff_back_input_space = Box(-1, 1, (ff_back_cfg.hidden_layers[0],))

        # Create LSTM
        lstm_cfg = concat_lstm_config.lstm
        lstm_individual_size = LSTMNumPy.get_individual_size(lstm_cfg, ff_front_output_space, ff_back_input_space)

        lstm_individual = individual[current_index:current_index + lstm_individual_size]
        current_index += lstm_individual_size

        lstm = LSTMNumPy(ff_front_output_space, ff_back_input_space, lstm_individual, lstm_cfg)

        # Create Feed Forward at the back here
        ff_back_individual_size = FeedForwardNumPy.get_individual_size(ff_back_cfg, ff_back_input_space, output_space)
        ff_back_individual = individual[current_index:current_index + ff_back_individual_size]
        current_index += ff_back_individual_size
        feed_forward_back = FeedForwardNumPy(ff_back_input_space, output_space, ff_back_individual, ff_back_cfg)

        assert current_index == len(individual)

        # Hidden and cell states are random, initialize them to the same arrays
        hidden_concat = np.random.randn(*concatenated_lstm.lstm.hidden.shape)
        cell_concat = np.random.randn(*concatenated_lstm.lstm.cell_state.shape)

        hidden_single_step = hidden_concat.copy()
        cell_single_step = cell_concat.copy()

        concatenated_lstm.lstm.hidden = hidden_concat
        concatenated_lstm.lstm.cell_state = cell_concat

        lstm.hidden = hidden_single_step
        lstm.cell_state = cell_single_step

        # Construct random input and compare the results
        random_input_concat = np.random.randn(input_size)
        random_input_single_steps = np.copy(random_input_concat)

        output_concat = concatenated_lstm.step(random_input_concat)

        x = feed_forward_front.step(random_input_single_steps)
        x = lstm.step(x)
        output_single_steps = feed_forward_back.step(x)

        assert np.allclose(output_concat, output_single_steps)
