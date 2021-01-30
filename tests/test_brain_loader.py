import brains.i_brain
import tools.configurations
import typing
import gym.spaces.discrete
import numpy as np
import pytest


@brains.i_brain.IBrain.register("TestBrain")
class BrainClass(brains.i_brain.IBrain[tools.configurations.IBrainCfg]):
    def __init__(self, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)

    def step(self, *nargs, **kwargs):
        return 1

    @classmethod
    def get_individual_size(cls, *nargs, **kwargs):
        return 1


class TestBrainLoader:

    def test_basic_init(self):
        input_space = gym.spaces.discrete.Discrete(1)
        output_space = gym.spaces.discrete.Discrete(1)
        config = tools.configurations.IBrainCfg(type='TestBrain')
        brain_class: typing.Type[brains.i_brain.IBrain] = brains.i_brain.registered_brain_classes[config.type]

        individual = np.zeros(
            brain_class.get_individual_size(input_space=input_space, output_space=output_space, config=config))
        brain = brain_class(input_space=input_space, output_space=output_space, individual=individual, config=config)
        assert 1 == brain.step(input_space.sample())

    def test_unregistered_type(self):
        config = tools.configurations.IBrainCfg(type='Smooth')
        with pytest.raises(KeyError, match='Smooth'):
            brain_class: typing.Type[brains.i_brain.IBrain] = brains.i_brain.registered_brain_classes[config.type]

    def test_duplicated_registering(self):
        with pytest.raises(AssertionError, match='TestBrain'):
            brains.i_brain.IBrain.register("TestBrain")(BrainClass)
