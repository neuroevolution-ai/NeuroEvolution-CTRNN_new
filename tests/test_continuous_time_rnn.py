from brains.continuous_time_rnn import ContinuousTimeRNN
import numpy as np
from gym.spaces import Box
import pytest
from attr import evolve, s


@s(auto_attribs=True, frozen=True, slots=True)
class BrainParam:
    V: np.ndarray
    W: np.ndarray
    T: np.ndarray
    y0: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray


class TestCTRNN:

    @pytest.fixture
    def brain_param_simple(self):
        return BrainParam(
            V=np.array([[0, 1],
                        [2, 3]]),
            W=np.array([[4, 5],
                        [6, 7]]),
            T=np.array([[8, 9],
                        [10, 11]]),
            y0=np.array([12, 13]),
            clip_min=np.array([14, 15]),
            clip_max=np.array([16, 17]))

    @pytest.fixture
    def brain_param_identity(self):
        return BrainParam(
            V=np.eye(2),
            W=np.eye(2),
            T=np.eye(2),
            y0=np.zeros(2),
            clip_min=np.array([-10, -15]),
            clip_max=np.array([10, 15]))

    @staticmethod
    def param_to_genom(param):
        return np.concatenate(
            [param.V.flatten(),
             param.W.flatten(),
             param.T.flatten(),
             param.y0.flatten(),
             param.clip_min.flatten(),
             param.clip_max.flatten()])

    def test_individual(self, ctrnn_config, brain_param_simple, box2d):
        ctrnn_config = evolve(ctrnn_config, set_principle_diagonal_elements_of_W_negative=False)

        ContinuousTimeRNN.generate_and_set_class_state(config=ctrnn_config, input_space=box2d,
                                                       output_space=Box(-1, 1, shape=[2]), )
        bp = brain_param_simple
        brain = ContinuousTimeRNN(input_space=box2d, output_space=box2d, individual=self.param_to_genom(bp),
                                  config=ctrnn_config)
        assert np.array_equal(bp.V, brain.V.toarray())
        assert np.array_equal(bp.W, brain.W.toarray())
        assert np.array_equal(bp.T, brain.T.toarray())
        assert np.array_equal(bp.y0, brain.y0)
        assert np.array_equal(bp.y0, brain.y)

    def test_step(self, ctrnn_config, brain_param_identity, box2d):
        ctrnn_config = evolve(ctrnn_config, set_principle_diagonal_elements_of_W_negative=False)
        bp = brain_param_identity
        ContinuousTimeRNN.generate_and_set_class_state(config=ctrnn_config, input_space=box2d,
                                                       output_space=Box(-1, 1, shape=[2]), )
        brain = ContinuousTimeRNN(input_space=box2d, output_space=box2d, individual=self.param_to_genom(bp),
                                  config=ctrnn_config)
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

    def test_clipping_legacy(self, ctrnn_config, brain_param_identity, box2d):
        bp = brain_param_identity
        ContinuousTimeRNN.generate_and_set_class_state(config=ctrnn_config, input_space=box2d,
                                                       output_space=Box(-1, 1, shape=[2]), )
        brain = ContinuousTimeRNN(input_space=box2d, output_space=box2d, individual=self.param_to_genom(bp),
                                  config=ctrnn_config)
        ob = np.array([1, 1])
        res = brain.step(ob * 1000)
        # due to tanh the maximum output is 1.0
        assert np.allclose(res, np.ones(2))
        # with legacy-clipping everything is clipped to the lowest max-value, which is 10 in this genome
        assert np.allclose(brain.y, np.ones(2) * 10)

    def test_clipping_per_neuron(self, ctrnn_config, brain_param_identity, box2d):
        ctrnn_config = evolve(ctrnn_config, optimize_state_boundaries="per_neuron")
        bp = evolve(brain_param_identity, clip_max=np.array([2, 3]), clip_min=np.array([-4, -5]))

        ContinuousTimeRNN.generate_and_set_class_state(config=ctrnn_config, input_space=box2d,
                                                       output_space=box2d, )
        brain = ContinuousTimeRNN(input_space=box2d, output_space=box2d,
                                  individual=self.param_to_genom(bp), config=ctrnn_config)
        ob = np.array([1, 1])
        brain.step(ob * 100000)
        assert np.allclose(brain.y, bp.clip_max)
        brain.step(ob * -100000)
        assert np.allclose(brain.y, bp.clip_min)

    def test_generating_masks_random(self, ctrnn_config, box2d):
        ContinuousTimeRNN.generate_and_set_class_state(
            config=evolve(ctrnn_config, v_mask="random", t_mask="random", w_mask="random", v_mask_param=0.5,
                          w_mask_param=0.5, t_mask_param=0.5), input_space=box2d,
            output_space=box2d, )

    def test_generating_masks_dense(self, ctrnn_config, box2d):
        ContinuousTimeRNN.generate_and_set_class_state(
            config=evolve(ctrnn_config, v_mask="dense", t_mask="dense", w_mask="dense"), input_space=box2d,
            output_space=box2d, )

    def test_generating_masks_log(self, ctrnn_config, box2d):
        ContinuousTimeRNN.generate_and_set_class_state(
            config=evolve(ctrnn_config, v_mask="logarithmic", w_mask="logarithmic", t_mask="logarithmic",
                          v_mask_param=2.5, w_mask_param=5, t_mask_param=1.1),
            input_space=box2d,
            output_space=box2d, )

    def test_wrong_size(self, ctrnn_config, box2d, brain_param_identity):
        ContinuousTimeRNN.generate_and_set_class_state(config=ctrnn_config, input_space=box2d,
                                                       output_space=box2d, )
        ind = self.param_to_genom(brain_param_identity)
        ContinuousTimeRNN(input_space=box2d, output_space=box2d,
                          individual=np.append(ind[:-1], [1]), config=ctrnn_config)
        with pytest.raises(AssertionError):
            ContinuousTimeRNN(input_space=box2d, output_space=box2d,
                              individual=ind[:-1], config=ctrnn_config)

    def test_get_individual_size(self, ctrnn_config):
        in_space = Box(-1, 1, shape=[3])
        out_space = Box(-1, 1, shape=[3])
        ContinuousTimeRNN.generate_and_set_class_state(config=ctrnn_config, input_space=in_space,
                                                       output_space=out_space, )
        ind_size = ContinuousTimeRNN.get_individual_size(config=ctrnn_config, input_space=in_space,
                                                         output_space=out_space, )
        assert ind_size == 22
