from gym.spaces import Space, Box
import numpy as np

from brains.i_brain import IBrain
from brains.lstm import LSTMNumPy
from brains.ffnn import FeedForwardNumPy
from tools.configurations import ConcatenatedBrainLSTMCfg, LSTMCfg, FeedForwardCfg


class ConcatenatedLSTM(IBrain):
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray,
                 config: ConcatenatedBrainLSTMCfg):
        super().__init__(input_space, output_space, individual, config)

        self.config = config
        self.input_space = input_space
        self.output_space = output_space
        self.feed_forward_front = None
        self.feed_forward_back = None

        feed_forward_front_cfg, feed_forward_back_cfg, lstm_config, lstm_input_size, lstm_output_size = (
            self.get_configs_and_output_sizes(self.config, input_space, output_space))

        current_index = 0

        if config.feed_forward_front:
            ff_front_individual_size = FeedForwardNumPy.get_individual_size(feed_forward_front_cfg, input_space,
                                                                            Box(-1, 1, (lstm_input_size,)))

            self.feed_forward_front = FeedForwardNumPy(input_space, Box(-1, 1, (lstm_input_size,)),
                                                       individual[
                                                       current_index:current_index + ff_front_individual_size],
                                                       feed_forward_front_cfg)
            current_index += ff_front_individual_size

        lstm_individual_size = LSTMNumPy.get_individual_size(lstm_config, Box(-1, 1, (lstm_input_size,)),
                                                             Box(-1, 1, (lstm_output_size,)))

        self.lstm = LSTMNumPy(Box(-1, 1, (lstm_input_size,)), Box(-1, 1, (lstm_output_size,)),
                              individual[current_index:current_index + lstm_individual_size], lstm_config)

        current_index += lstm_individual_size

        if config.feed_forward_back:
            ff_back_individual_size = FeedForwardNumPy.get_individual_size(feed_forward_back_cfg,
                                                                           Box(-1, 1, (lstm_output_size,)),
                                                                           output_space)

            self.feed_forward_back = FeedForwardNumPy(Box(-1, 1, (lstm_output_size,)), output_space,
                                                      individual[current_index:current_index + ff_back_individual_size],
                                                      feed_forward_back_cfg)

            current_index += ff_back_individual_size

        assert current_index == len(individual)

    def brainstep(self, ob: np.ndarray):
        x = ob

        if self.feed_forward_front:
            x = self.feed_forward_front.step(x)

        x = self.lstm.step(x)

        if self.feed_forward_back:
            x = self.feed_forward_back.step(x)

        return x

    @classmethod
    def get_configs_and_output_sizes(cls, config: ConcatenatedBrainLSTMCfg, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        lstm_input_size = input_size
        lstm_output_size = output_size

        feed_forward_front_cfg = None
        feed_forward_back_cfg = None
        lstm_config = config.lstm if isinstance(config.lstm, LSTMCfg) else LSTMCfg(**config.lstm)

        if config.feed_forward_front:
            feed_forward_front_cfg = (
                config.feed_forward_front if isinstance(config.feed_forward_front, FeedForwardCfg) else FeedForwardCfg(
                    **config.feed_forward_front))

            lstm_input_size = feed_forward_front_cfg.hidden_layers[-1]

        if config.feed_forward_back:
            feed_forward_back_cfg = (
                config.feed_forward_back if isinstance(config.feed_forward_back, FeedForwardCfg) else FeedForwardCfg(
                    **config.feed_forward_back))

            lstm_output_size = feed_forward_back_cfg.hidden_layers[0]

        return feed_forward_front_cfg, feed_forward_back_cfg, lstm_config, lstm_input_size, lstm_output_size

    @classmethod
    def get_individual_size(cls, config: ConcatenatedBrainLSTMCfg, input_space: Space, output_space: Space):

        feed_forward_front_cfg, feed_forward_back_cfg, lstm_config, lstm_input_size, lstm_output_size = (
            cls.get_configs_and_output_sizes(config, input_space, output_space))

        individual_size = 0

        if config.feed_forward_front:
            individual_size += FeedForwardNumPy.get_individual_size(feed_forward_front_cfg, input_space,
                                                                    Box(-1, 1, (lstm_input_size,)))

        individual_size += LSTMNumPy.get_individual_size(lstm_config, Box(-1, 1, (lstm_input_size,)),
                                                         Box(-1, 1, (lstm_output_size,)))

        if config.feed_forward_back:
            individual_size += FeedForwardNumPy.get_individual_size(feed_forward_back_cfg,
                                                                    Box(-1, 1, (lstm_output_size,)), output_space)

        return individual_size
