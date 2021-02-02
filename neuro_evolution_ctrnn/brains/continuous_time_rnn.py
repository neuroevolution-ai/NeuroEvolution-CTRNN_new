import numpy as np
from tools.configurations import ContinuousTimeRNNCfg
from typing import List, Union
from gym.spaces import Space, Box, Discrete
import logging
import math
from brains.i_brain import IBrain
from scipy import sparse


class ContinuousTimeRNN(IBrain[ContinuousTimeRNNCfg]):
    v_mask: np.ndarray
    w_mask: np.ndarray
    t_mask: np.ndarray

    @classmethod
    def get_class_state(cls):
        return {"v_mask": cls.v_mask, "w_mask": cls.w_mask, "t_mask": cls.t_mask}

    @classmethod
    def set_class_state(cls, v_mask, w_mask, t_mask):
        cls.v_mask = v_mask
        cls.w_mask = w_mask
        cls.t_mask = t_mask

    def learned_sparse(self, mat, percentile):
        M = np.abs(mat.toarray())
        p = np.percentile(a=M, q=percentile, interpolation='higher')
        mat[M < p] = 0
        return sparse.csr_matrix(mat, dtype=float)

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ContinuousTimeRNNCfg):
        super().__init__(input_space, output_space, individual, config)
        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        optimize_y0 = config.optimize_initial_neuron_state
        delta_t = config.delta_t
        self.config = config
        self.input_space: Space = input_space
        self.output_space: Space = output_space
        N_n = config.number_neurons

        self.delta_t = delta_t

        # insert weights-values into weight-masks to receive weight-matrices
        # explanation here: https://stackoverflow.com/a/61968524/5132456
        V_size: int = np.count_nonzero(self.v_mask)  # type: ignore
        W_size: int = np.count_nonzero(self.w_mask)  # type: ignore
        T_size: int = np.count_nonzero(self.t_mask)  # type: ignore
        self.V = sparse.csr_matrix(self.v_mask, dtype=float)
        self.W = sparse.csr_matrix(self.w_mask, dtype=float)
        self.T = sparse.csr_matrix(self.t_mask, dtype=float)
        self.V.data[:] = [element for element in individual[0:V_size]]
        self.W.data[:] = [element for element in individual[V_size:V_size + W_size]]
        self.T.data[:] = [element for element in individual[V_size + W_size:V_size + W_size + T_size]]

        if self.config.v_mask == 'learned':
            self.V = self.learned_sparse(self.V, self.config.v_mask_param)

        if self.config.w_mask == 'learned':
            self.W = self.learned_sparse(self.W, self.config.w_mask_param)

        if self.config.t_mask == 'learned':
            self.T = self.learned_sparse(self.T, self.config.t_mask_param)

        index: int = V_size + W_size + T_size

        # Initial state values y0
        if optimize_y0:
            self.y0 = np.array([element for element in individual[index:index + N_n]])
            index += N_n
        else:
            self.y0 = np.zeros(N_n)

        self.y = self.y0

        # Clipping ranges for state boundaries
        self.clipping_range_min: Union[int, np.ndarray, List[int]]
        self.clipping_range_max: Union[int, np.ndarray, List[int]]
        if self.config.optimize_state_boundaries == "per_neuron":
            self.clipping_range_min = np.asarray([-abs(element) for element in individual[index:index + N_n]])
            self.clipping_range_max = np.asarray([abs(element) for element in individual[index + N_n:]])
        elif self.config.optimize_state_boundaries == "global":
            self.clipping_range_min = -abs(individual[index])
            self.clipping_range_max = abs(individual[index + 1])
        elif self.config.optimize_state_boundaries == "legacy":
            self.clipping_range_min = [-abs(element) for element in individual[index:index + N_n]]
            self.clipping_range_max = [abs(element) for element in individual[index + N_n:]]
        elif self.config.optimize_state_boundaries == "fixed":
            self.clipping_range_min = np.asarray([config.clipping_range_min] * N_n)
            self.clipping_range_max = np.asarray([config.clipping_range_max] * N_n)
        else:
            raise RuntimeError("unknown parameter for optimize_state_boundaries")

        # Set elements of main diagonal to less than 0
        if config.set_principle_diagonal_elements_of_W_negative:
            for j in range(N_n):
                if self.W[j, j]:  # this if is a speedup when dealing with sparse matrices
                    self.W[j, j] = -abs(self.W[j, j])

    def brainstep(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:
        if isinstance(self.input_space, Discrete):
            # todo: do this observation-transformation in a cleaner way
            ob = self.discrete_to_vector(ob)
        else:
            # RGB-Data usually comes in 210x160x3 shape, but V is always 1D-Vector
            ob = ob.flatten()

        if self.config.use_bias:
            ob = np.r_[ob, [1]]

        # Differential equation
        if self.config.neuron_activation == "relu":
            y_ = np.maximum(0, self.y)
        elif self.config.neuron_activation == "tanh":
            y_ = np.tanh(self.y)
        elif self.config.neuron_activation == "learned":
            # value = alpha * np.tanh(self.y) + (1-alpha) * np.relu(self.y)
            raise NotImplementedError("learned activations are not yet implemented")
        else:
            raise RuntimeError("unknown aktivation function: " + str(self.config.neuron_activation))

        if self.config.neuron_activation_inplace:
            self.y = y_  # type:ignore
        dydt: np.ndarray = self.W.dot(y_) + self.V.dot(ob)

        # Euler forward discretization
        self.y = self.y + self.delta_t * dydt

        if self.config.parameter_perturbations:
            self.y += np.random.normal([0] * len(self.y), self.config.parameter_perturbations)

        if self.config.optimize_state_boundaries == "legacy":
            for y_min, y_max in zip(self.clipping_range_min, self.clipping_range_max):  # type: ignore
                self.y = np.clip(self.y, y_min, y_max)
        else:
            self.y = np.clip(self.y, self.clipping_range_min, self.clipping_range_max)

        o: Union[np.ndarray, np.generic] = np.tanh(self.T.dot(self.y))
        return o

    @classmethod
    def get_individual_size(cls, config: ContinuousTimeRNNCfg, input_space: Space, output_space: Space):
        individual_size = np.count_nonzero(cls.v_mask) + np.count_nonzero(cls.w_mask) + np.count_nonzero(cls.t_mask)
        if config.optimize_initial_neuron_state:
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
    def set_masks_globally(cls, config: ContinuousTimeRNNCfg, input_space: Space, output_space: Space):
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)

        if hasattr(cls, "v_mask") or hasattr(cls, "w_mask") or hasattr(cls, "t_mask"):
            logging.warning("masks are already present in class")
        # todo: also store masks in checkpoints and hof.
        v_mask = cls._generate_mask(config.v_mask, config.number_neurons, input_size, config.v_mask_param)
        if config.use_bias:
            v_mask = np.c_[v_mask, np.ones(config.number_neurons, dtype=bool)]

        w_mask = cls._generate_mask(config.w_mask, config.number_neurons, config.number_neurons,
                                    config.w_mask_param)
        # TODO The mask was flipped on the diagonal for mathematical correct structure. check if no errer exist
        t_mask = cls._generate_mask(config.t_mask, output_size, config.number_neurons, config.t_mask_param)

        cls.set_class_state(v_mask=v_mask, w_mask=w_mask, t_mask=t_mask)

    @staticmethod
    def _generate_mask(mask_type, n, m, mask_param):
        if mask_type == "random":
            return np.random.rand(n, m) < mask_param
        elif mask_type == "logarithmic":
            if mask_param < 1.05:
                raise RuntimeError("mask_param to small: " + str(mask_param) + " must be at least +1.05")
            base = mask_param
            indices = [math.floor(base ** y) for y in np.arange(0, math.floor(math.log(max(m, n), base)) + 1, 1)]
            indices = indices
            result = np.zeros((n, m), dtype=bool)

            # when the matrix is rectangular, we need to fine a pseudo-diagonal instead an
            # actual diagonal, to guarantee each row and column gets values
            if m > n:
                stretch = np.rint(np.array(range(m)) * ((n - 1) / m)).astype(int)
                pseudo_diag = zip(stretch, range(m))
            else:
                stretch = np.rint(np.array(range(n)) * ((m - 1) / n)).astype(int)
                pseudo_diag = zip(range(n), stretch)
            for x, y in pseudo_diag:
                for j in indices:
                    # set value left of diagonal
                    result[x][(y + j) % m] = True
                    # set value right of diagonal
                    result[x][(y - j) % m] = True

            return result

        elif mask_type == "dense":
            return np.ones((n, m), dtype=bool)
        elif mask_type == "learned":
            return np.ones((n, m), dtype=bool)
        else:
            raise RuntimeError("unknown mask_type: " + str(mask_type))

    def get_neuron_states(self):
        return self.y

    def get_internal_weight_matrix(self):
        return self.W.todense()

    def get_input_weight_matrix(self):
        return self.V.toarray()

    def get_output_weight_matrix(self):
        return self.T.toarray()
