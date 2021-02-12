import brains.i_brain
import tools.configurations
import typing
import gym.spaces.discrete
import numpy as np
import pytest


class BrainCfgClass(tools.configurations.IBrainCfg):
    pass


class BrainCfgClassDuplicate(tools.configurations.IBrainCfg):
    pass


@brains.i_brain.register_brain("TestBrain", BrainCfgClass)
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
        input_space = gym.spaces.discrete.Discrete(1)
        output_space = gym.spaces.discrete.Discrete(1)
        config = BrainCfgClass(type='TestBrain')

        brain_class: typing.Type[
            brains.i_brain.IBrain] = brains.i_brain.registered_brains[
                config.type]

        individual = np.zeros(
            brain_class.get_individual_size(input_space=input_space,
                                            output_space=output_space,
                                            config=config))
        brain = brain_class(input_space=input_space,
                            output_space=output_space,
                            individual=individual,
                            config=config)

        # Needs to be zero because the Discrete Space has n set to 1 so only zeros can be used as input
        assert 0 == brain.step(input_space.sample())

    def test_two_brains_for_one_config(self):
        brains.i_brain.register_brain("TestBrainDupe",
                                      BrainCfgClassDuplicate)(BrainClassDupe1)
        with pytest.raises(AssertionError, match='TestBrainDupe'):
            brains.i_brain.register_brain(
                "TestBrainDupe", BrainCfgClassDuplicate)(BrainClassDupe2)

    def test_unregistered_type(self):
        config = tools.configurations.IBrainCfg(type='Smooth')
        with pytest.raises(KeyError, match='Smooth'):
            brain_class: typing.Type[
                brains.i_brain.IBrain] = brains.i_brain.registered_brains[
                    config.type]
