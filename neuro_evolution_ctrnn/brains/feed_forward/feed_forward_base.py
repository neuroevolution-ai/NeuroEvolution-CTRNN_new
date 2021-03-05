from typing import List, TypeVar

import numpy as np
from gym.spaces import Space, flatdim

from brains.brain import Brain
from tools.configurations import FeedForwardCfg

ConfigClass = TypeVar("ConfigClass", bound=FeedForwardCfg)


class FeedForwardBase(Brain[ConfigClass]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ConfigClass):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config=config, input_space=input_space,
                                                           output_space=output_space)

        self.input_size: int = flatdim(input_space)
        self.output_size: int = flatdim(output_space)
        self.config = config
        self.use_bias = config.use_bias

        # If the check fails the program aborts
        self.hidden_layers: List[int] = self.check_hidden_layers(config.hidden_layers)

    def brainstep(self, ob: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def check_hidden_layers(hidden_layers: List[int]) -> List[int]:
        try:
            assert len(hidden_layers) > 0

            # Design Space, unpack first
            if any(isinstance(i, list) for i in hidden_layers):
                # Simply take the first list, should be the only one. If not somewhere the decision of the Design Space
                # did not occur
                hidden_layers = hidden_layers[0]
            assert all(hidden_layers) > 0
            return hidden_layers
        except AssertionError:
            raise RuntimeError(
                "Error with the chosen hidden layer list {}. It must be at least of size 1 and have elements which are "
                "larger than 0.".format(hidden_layers))

    @classmethod
    def get_free_parameter_usage(cls, config: ConfigClass, input_space: Space, output_space: Space):
        input_size = flatdim(input_space)
        output_size = flatdim(output_space)

        hidden_layers = cls.check_hidden_layers(config.hidden_layers)

        individual_size = 0
        last_layer = input_size

        for hidden_layer in hidden_layers:
            individual_size += last_layer * hidden_layer
            last_layer = hidden_layer

        individual_size += last_layer * output_size

        if config.use_bias:
            individual_size += sum(hidden_layers) + output_size

        return {"individual_size": individual_size}
