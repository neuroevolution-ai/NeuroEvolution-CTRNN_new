import brains.i_brain
import tools.configurations
import typing
import gym.spaces.discrete
import numpy as np
import pytest
from tools.helper import brain_class_from_config_type


@tools.configurations.register('TestBrain')
class BrainCfgClass(tools.configurations.IBrainCfg):
    pass


class BrainClass(brains.i_brain.IBrain[BrainCfgClass]):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    def step(self, *nargs, **kwargs):
        return 1

    @classmethod
    def get_free_parameter_usage(cls, *nargs, **kwargs):
        return {"blubb": 1}


class TestBrainLoader:

    def test_basic_init(self):
        input_space = gym.spaces.discrete.Discrete(1)
        output_space = gym.spaces.discrete.Discrete(1)
        config = BrainCfgClass(type='TestBrain')

        brain_class: typing.Type[brains.i_brain.IBrain] = brain_class_from_config_type(config.type)

        individual = np.zeros(
            brain_class.get_individual_size(input_space=input_space, output_space=output_space, config=config))
        brain = brain_class(input_space=input_space, output_space=output_space, individual=individual, config=config)
        assert 1 == brain.step(input_space.sample())

    def test_unregistered_type(self):
        config = tools.configurations.IBrainCfg(type='Smooth')
        with pytest.raises(KeyError, match='Smooth'):
            brain_class: typing.Type[brains.i_brain.IBrain] = brain_class_from_config_type(config.type)
