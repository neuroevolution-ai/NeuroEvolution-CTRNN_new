import numpy as np
from scipy.ndimage import convolve
import torch
import time
import pytest


@pytest.mark.skip(reason="General Testing, not a test case of the program itself")
class TestConvolutionTimes:

    def setup_method(self):
        self.input_size = (1, 3, 64, 64)

        self.input_torch = np.random.randn(*self.input_size)
        self.input_scipy = np.copy(self.input_torch)

    def test_single_convolution_times(self):
        scipy_kernel = np.random.randn(5, 3, 3, 3)

        conv_torch = torch.nn.Conv2d(3, 5, 3, stride=1, padding=0, bias=False)
        conv_torch.weight.data = torch.from_numpy(np.copy(scipy_kernel))

        assert np.array_equal(self.input_torch, self.input_scipy)

        times_scipy = []
        times_torch = []

        for _ in range(1000):
            time_s = time.time()
            result_scipy = convolve(self.input_scipy, scipy_kernel)
            times_scipy.append(time.time() - time_s)

        for _ in range(1000):
            time_s = time.time()
            result_torch = conv_torch(torch.from_numpy(self.input_torch))
            times_torch.append(time.time() - time_s)

        print("Scipy Mean {}".format(np.mean(times_scipy)))
        print("Scipy Std {}".format(np.std(times_scipy)))
        print("Scipy Max {}".format(np.max(times_scipy)))
        print("Scipy Min {}".format(np.min(times_scipy)))

        print("-----------------")

        print("Torch Mean {}".format(np.mean(times_torch)))
        print("Torch Std {}".format(np.std(times_torch)))
        print("Torch Max {}".format(np.max(times_torch)))
        print("Torch Min {}".format(np.min(times_torch)))

        print(result_scipy)
        print("---- TORCH:--------")
        print(result_torch)


