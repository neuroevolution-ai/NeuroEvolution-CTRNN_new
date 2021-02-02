import pytest
from attr import s
import numpy as np
from gym import Space
from gym.spaces import Box

from brains.i_layer_based_brain import ILayerBasedBrain, LayerdConfigClass
from tools.configurations import ILayerBasedBrainCfg


@s(auto_attribs=True, frozen=True, slots=True)
class BrainParam:
    weight_ih: np.ndarray
    weight_hh: np.ndarray
    bias_h: np.ndarray
    hidden: np.ndarray
    weight_ho: np.ndarray


class LayerTestCase(ILayerBasedBrain[ILayerBasedBrainCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LayerdConfigClass):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def get_number_gates():
        return 1

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        result = np.dot(weight_ih[0], layer_input) + np.dot(weight_hh[0], hidden) + bias_h[0]
        return [result, result]


class TestILayeredBrain:

    @pytest.fixture
    def brain_param_simple(self):
        return BrainParam(
            weight_ih=np.array([[[[1, 2], [3, 4]]]]),
            weight_hh=np.array([[[[5, 6], [7, 8]]]]),
            bias_h=np.array([[[9, 10]]]),
            hidden=np.array([[11, 12]]),
            weight_ho=np.array([[13, 14], [15, 16]])
        )

    @pytest.fixture
    def brain_param_identity(self):
        return BrainParam(
            weight_ih=np.array([[[[1, 0], [0, 1]]]]),
            weight_hh=np.array([[[[1, 0], [0, 1]]]]),
            bias_h=np.array([[[0, 0]]]),
            hidden=np.array([[0, 0]]),
            weight_ho=np.array([[1, 0], [0, 1]])
        )

    @staticmethod
    def param_to_genom(param):
        return np.concatenate(
            [param.weight_ih.flatten(),
             param.weight_hh.flatten(),
             param.bias_h.flatten(),
             param.hidden.flatten(),
             param.weight_ho.flatten()
             ])

    def test_individual(self, layer_config, brain_param_simple, box2d):
        bp = brain_param_simple
        brain = LayerTestCase(input_space=box2d, output_space=box2d, individual=self.param_to_genom(bp),
                              config=layer_config)
        assert np.array_equal(bp.weight_ih, brain.weight_ih)
        assert np.array_equal(bp.weight_hh, brain.weight_hh)
        assert np.array_equal(bp.bias_h, brain.bias_h)
        assert np.array_equal(bp.hidden, brain.hidden)
        assert np.array_equal(bp.weight_ho, brain.weight_ho)

    def test_step(self, layer_config, brain_param_identity, box2d):
        bp = brain_param_identity
        brain = LayerTestCase(input_space=box2d, output_space=box2d, individual=self.param_to_genom(bp),
                              config=layer_config)
        ob = np.array([1, 1])
        assert np.allclose(brain.hidden, np.zeros([2, 2]))
        res = brain.step(ob)
        # due to identity matrices after one iteration the internal state is now exactly the observersion
        assert np.allclose(brain.hidden, ob)
        # due to identity matrices after one iteration the output is just the input, but with tanh.
        assert np.allclose(res, ob)
        brain.step(ob)
        assert np.allclose(brain.hidden, ob + ob)
