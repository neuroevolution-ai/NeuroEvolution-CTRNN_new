from brains.layered_nn import LayeredNN, LayeredNNCfg
import numpy as np


class TestLayerNN:

    def test_init_and_step(self, lnn_config, box2d):
        size = LayeredNN.get_individual_size(input_space=box2d, output_space=box2d,
                                             config=lnn_config)
        brain = LayeredNN(input_space=box2d, output_space=box2d,
                          config=lnn_config, individual=range(size))
        ob = np.array([1, 1])
        res = brain.step(ob)
        assert np.allclose(res, np.array([1, 1]))
