# import neu

from brains.continuous_time_rnn import ContinuousTimeRNNCfg, ContinuousTimeRNN
import numpy as np
from collections import namedtuple
from gym.spaces import Box
import copy


class TestCTRNN:
    c = ContinuousTimeRNNCfg(optimize_y0=True,
                             normalize_input=False,
                             normalize_input_target=0,
                             delta_t=0.05,
                             optimize_state_boundaries="legacy",
                             set_principle_diagonal_elements_of_W_negative=False,
                             number_neurons=2,
                             clipping_range_min=1.0,
                             clipping_range_max=1.0)
    brain_param = namedtuple("brain_param", ["V", "W", "T", "y0", "clip_min", "clip_max"])
    brain_param_simple = brain_param(
        V=np.array([[0, 1],
                    [2, 3]]),
        W=np.array([[4, 5],
                    [6, 7]]),
        T=np.array([[8, 9],
                    [10, 11]]),
        y0=np.array([12, 13]),
        clip_min=np.array([14, 15]),
        clip_max=np.array([16, 17]))

    box2d = Box(-1, 1, shape=[2])

    brain_param_identity = brain_param(
        V=np.eye(2),
        W=np.eye(2),
        T=np.eye(2),
        y0=np.zeros(2),
        clip_min=np.array([-10, -15]),
        clip_max=np.array([10, 15]))

    def param_to_genom(self, brain_param):
        return np.concatenate(
            [brain_param.V.flatten(),
             brain_param.W.flatten(),
             brain_param.T.flatten(),
             brain_param.y0.flatten(),
             brain_param.clip_min.flatten(),
             brain_param.clip_max.flatten()])

    def test_individual(self):
        #     def __init__(self, low, high, shape=None, dtype=np.float32):

        bp = self.brain_param_simple
        brain = ContinuousTimeRNN(input_space=self.box2d, output_size=2, individual=self.param_to_genom(bp),
                                  config=self.c)
        assert np.array_equal(bp.V, brain.V)
        assert np.array_equal(bp.W, brain.W)
        assert np.array_equal(bp.T, brain.T)
        assert np.array_equal(bp.y0, brain.y0)
        # todo y currently has wrong shape. This should work without flattening it first
        assert np.array_equal(bp.y0, brain.y.flatten())

    def test_step(self):
        bp = self.brain_param_identity
        brain = ContinuousTimeRNN(input_space=self.box2d, output_size=2, individual=self.param_to_genom(bp),
                                  config=self.c)
        brain.delta_t = 1.0
        ob = np.array([1, 1])
        assert np.allclose(brain.y, np.zeros([2, 2]))
        res = brain.step(ob)
        # due to identity matrices after one iteration the internal state is now exactly the observersion
        assert np.allclose(brain.y, ob)
        # due to identity matrices after one iteration the output is just the input, but with tanh.
        assert np.allclose(res, np.tanh(ob))
        brain.step(ob)
        assert np.allclose(brain.y, np.tanh(ob) + ob + ob)

    def test_clipping_legacy(self):
        bp = self.brain_param_identity
        brain = ContinuousTimeRNN(input_space=self.box2d, output_size=2, individual=self.param_to_genom(bp),
                                  config=self.c)
        ob = np.array([1, 1])
        res = brain.step(ob * 1000)
        # due to tanh the maximum output is 1.0
        assert np.allclose(res, np.ones(2))
        # with legacy-clipping everything is clipped to the lowest max-value, which is 10 in this genome
        assert np.allclose(brain.y, np.ones([2, 2]) * 10)

    def test_clipping_per_neuron(self):
        config = copy.deepcopy(self.c)
        config.optimize_state_boundaries = "per_neuron"
        y = self.brain_param_identity._asdict()
        y["clip_max"] = np.array([2, 3])
        y["clip_min"] = np.array([-4, -5])
        bp = self.brain_param(**y)
        brain = ContinuousTimeRNN(input_space=self.box2d, output_size=2,
                                  individual=self.param_to_genom(bp), config=config)
        ob = np.array([1, 1])
        brain.step(ob * 100000)
        assert np.allclose(brain.y, bp.clip_max)
        brain.step(ob * -100000)
        assert np.allclose(brain.y, bp.clip_min)

    def test__get_size_from_shape(self):
        size = ContinuousTimeRNN._get_size_from_shape([3, 3, 3])
        assert size == 27

    def test_get_individual_size(self):
        ind_size = ContinuousTimeRNN.get_individual_size(input_space=Box(-1, 1, shape=[3]), output_size=3,
                                                         config=self.c)
        assert ind_size == 22
