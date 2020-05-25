import numpy as np
from tools.configurations import ContinuousTimeRNNCfg
from typing import Any, Collection, List, Union
from gym.spaces import Space, Box
import logging


class ContinuousTimeRNN:
    v_mask: np.ndarray
    w_mask: np.ndarray
    t_mask: np.ndarray

    def __init__(self, input_space: Space, output_size: int, individual: np.ndarray, config: ContinuousTimeRNNCfg):
        assert len(individual) == self.get_individual_size(config)
        optimize_y0 = config.optimize_y0
        delta_t = config.delta_t
        self.config = config
        self.input_space: Space = input_space
        set_principle_diagonal_elements_of_W_negative = config.set_principle_diagonal_elements_of_W_negative
        N_n = config.number_neurons

        self.delta_t = delta_t
        self.set_principle_diagonal_elements_of_W_negative = set_principle_diagonal_elements_of_W_negative

        # insert weights-values into weight-masks to receive weight-matrices
        # explanation here: https://stackoverflow.com/a/61968524/5132456
        V_size: int = np.count_nonzero(self.v_mask)  # type: ignore
        W_size: int = np.count_nonzero(self.w_mask)  # type: ignore
        T_size: int = np.count_nonzero(self.t_mask)  # type: ignore
        self.V = np.zeros(self.v_mask.shape, float)
        self.W = np.zeros(self.w_mask.shape, float)
        self.T = np.zeros(self.t_mask.shape, float)
        self.V[self.v_mask] = [element for element in individual[0:V_size]]
        self.W[self.w_mask] = [element for element in individual[V_size:V_size + W_size]]
        self.T[self.t_mask] = [element for element in individual[V_size + W_size:V_size + W_size + T_size]]

        index: int = V_size + W_size + T_size

        # Initial state values y0
        if optimize_y0:
            self.y0 = np.array([element for element in individual[index:index + N_n]])
            index += N_n
        else:
            self.y0 = np.zeros(N_n)

        self.y = self.y0

        # Clipping ranges for state boundaries
        self.clipping_range_min: Union[int, np.ndarray]
        self.clipping_range_max: Union[int, np.ndarray]
        if self.config.optimize_state_boundaries == "per_neuron":
            self.clipping_range_min = np.asarray([-abs(element) for element in individual[index:index + N_n]])
            self.clipping_range_max = np.asarray([abs(element) for element in individual[index + N_n:]])
        elif self.config.optimize_state_boundaries == "global":
            self.clipping_range_min = -abs(individual[index])
            self.clipping_range_max = abs(individual[index+1])
        elif self.config.optimize_state_boundaries == "legacy":
            self.clipping_range_min = [-abs(element) for element in individual[index:index + N_n]]
            self.clipping_range_max = [abs(element) for element in individual[index + N_n:]]
        elif self.config.optimize_state_boundaries == "fixed":
            self.clipping_range_min = np.asarray([config.clipping_range_min] * N_n)
            self.clipping_range_max = np.asarray([config.clipping_range_max] * N_n)
        else:
            raise RuntimeError("unkown parameter for optimize_state_boundaries")

        # Set elements of main diagonal to less than 0
        if set_principle_diagonal_elements_of_W_negative:
            for j in range(N_n):
                self.W[j][j] = -abs(self.W[j][j])

    @staticmethod
    def _normalize(x, low, high):
        if high > 1e5 or low < -1e5:
            # treat high spaces as unbounded
            # note: some spaces some envs don't define input_space.bounded_below properly
            return x
        return (((x - low) / (high - low)) * 2) - 1

    def step(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:

        if self.config.normalize_input:
            for idx, item in enumerate(ob):
                if isinstance(self.input_space, Box):
                    if self.input_space.bounded_below[idx] and self.input_space.bounded_above[idx]:
                        ob[idx] = self._normalize(ob[idx], self.input_space.low[idx],
                                                  self.input_space.high[idx]) * self.config.normalize_input_target
                else:
                    raise NotImplementedError("normalize_input is only defined for input-type Box")

        # Differential equation
        dydt: np.ndarray = np.dot(self.W, np.tanh(self.y)) + np.dot(self.V, ob)

        # Euler forward discretization
        self.y = self.y + self.delta_t * dydt

        if self.config.parameter_perturbations:
            self.y = np.random.normal(self.y, self.config.parameter_perturbations)

        if self.config.optimize_state_boundaries == "legacy":
            for y_min, y_max in zip(self.clipping_range_min, self.clipping_range_max):
                self.y = np.clip(self.y, y_min, y_max)
        else:
            self.y = np.clip(self.y, self.clipping_range_min, self.clipping_range_max)

        o: Union[np.ndarray, np.generic] = np.tanh(np.dot(self.y, self.T))
        return o

    @staticmethod
    def _get_size_from_shape(shape: np.shape):
        size = 1
        for val in shape:
            size = size * val
        return size

    @classmethod
    def get_individual_size(cls, config: ContinuousTimeRNNCfg):
        individual_size = np.count_nonzero(cls.v_mask) + np.count_nonzero(cls.w_mask) + np.count_nonzero(cls.t_mask)
        if config.optimize_y0:
            individual_size += config.number_neurons

        if config.optimize_state_boundaries == "legacy":
            individual_size += 2 * config.number_neurons
        elif config.optimize_state_boundaries == "per_neuron":
            individual_size += 2 * config.number_neurons
        elif config.optimize_state_boundaries == "global":
            individual_size += 2
        elif config.optimize_state_boundaries == "fixed":
            individual_size += 0
        return individual_size

    @classmethod
    def set_masks_globally(cls, config: ContinuousTimeRNNCfg, input_space, output_space):
        input_size = cls._get_size_from_shape(input_space.shape)
        output_size = cls._get_size_from_shape(output_space.shape)

        if hasattr(cls, "v_mask") or hasattr(cls, "w_mask") or hasattr(cls, "t_mask"):
            logging.warning("masks are already present in class")

        cls.v_mask = cls._generate_mask(config.v_mask, config.number_neurons, input_size, config.v_mask_prob)
        cls.w_mask = cls._generate_mask(config.w_mask, config.number_neurons, config.number_neurons, config.w_mask_prob)
        cls.t_mask = cls._generate_mask(config.t_mask, config.number_neurons, output_size, config.t_mask_prob)

    @staticmethod
    def _generate_mask(mask_type, n, m, mask_prob):
        if mask_type == "random":
            return np.random.rand(n, m) < mask_prob
        elif mask_type == "dense":
            return np.ones((n, m), dtype=bool)
        else:
            raise RuntimeError("unknown mask_type: " + str(mask_type))
