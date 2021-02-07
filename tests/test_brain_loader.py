import typing
import gym.spaces.discrete
import numpy as np
import pytest
from tools.config_reader import registered_brains, register
import brains.i_brain
import tools.configurations


class BrainCfgClass(tools.configurations.IBrainCfg):
    pass


class BrainCfgClassDuplicate(tools.configurations.IBrainCfg):
    pass


class BrainClass(brains.i_brain.IBrain[BrainCfgClass]):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    def brainstep(self, *nargs, **kwargs):
        return 1

    @classmethod
    def get_free_parameter_usage(cls, *nargs, **kwargs):
        return {"blubb": 1}


class BrainClassDupe1(brains.i_brain.IBrain[BrainCfgClassDuplicate]):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    def brainstep(self, *nargs, **kwargs):
        return 1

    @classmethod
    def get_free_parameter_usage(cls, *nargs, **kwargs):
        return {"blubb": 1}


class BrainClassDupe2(brains.i_brain.IBrain[BrainCfgClassDuplicate]):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    def brainstep(self, *nargs, **kwargs):
        return 1

    @classmethod
    def get_free_parameter_usage(cls, *nargs, **kwargs):
        return {"blubb": 1}


class TestBrainLoader:

    def test_basic_init(self):
        register('TestBrain', brain_class=BrainClass)(BrainCfgClass)
        input_space = gym.spaces.discrete.Discrete(1)
        output_space = gym.spaces.discrete.Discrete(1)
        config = BrainCfgClass(type='TestBrain')

        brain_class: typing.Type[brains.i_brain.IBrain] = registered_brains[config.type]

        individual = np.zeros(
            brain_class.get_individual_size(input_space=input_space, output_space=output_space, config=config))
        brain = brain_class(input_space=input_space, output_space=output_space, individual=individual, config=config)
        assert 1 == brain.step(input_space.sample())

    def test_two_brains_for_one_config(self):
        config = tools.configurations.IBrainCfg(type='TestBrainDupe')
        with pytest.raises(AssertionError):
            register('TestBrainDupe', brain_class=BrainClassDupe1)(BrainCfgClassDuplicate)
            register('TestBrainDupe', brain_class=BrainClassDupe2)(BrainCfgClassDuplicate)

    def test_unregistered_type(self):
        config = tools.configurations.IBrainCfg(type='Smooth')
        with pytest.raises(KeyError, match='Smooth'):
            brain_class: typing.Type[brains.i_brain.IBrain] = registered_brains[config.type]
