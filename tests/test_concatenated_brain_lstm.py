from tools.configurations import ConcatenatedBrainLSTMCfg
from brains.concatenated_brains import ConcatenatedLSTM

import numpy as np
from gym.spaces import Box


class TestConcatenatedLSTM:

    def test_concatenated_lstm_output(self, concat_lstm_config: ConcatenatedBrainLSTMCfg):
        input_space = Box(-1, 1, (28,))
        output_space = Box(-1, 1, (8,))

        individual_size = ConcatenatedLSTM.get_individual_size(concat_lstm_config, input_space, output_space)
        individual = np.random.randn(individual_size).astype(np.float32)

        concatenated_lstm = ConcatenatedLSTM(input_space, output_space, individual, concat_lstm_config)

        assert (concatenated_lstm.feed_forward_front.hidden_layers ==
                concat_lstm_config.feed_forward_front.hidden_layers[0])

        assert concatenated_lstm.lstm.input_space.shape[0] == concat_lstm_config.feed_forward_front.hidden_layers[0][-1]

        assert (concatenated_lstm.feed_forward_back.hidden_layers ==
                concat_lstm_config.feed_forward_back.hidden_layers[0])
