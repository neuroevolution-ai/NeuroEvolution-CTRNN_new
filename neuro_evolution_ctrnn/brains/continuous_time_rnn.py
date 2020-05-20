import numpy as np
from tools.configurations import ContinuousTimeRNNCfg


class ContinuousTimeRNN:

    def __init__(self, input_space: np.shape, output_size: int, individual: np.ndarray, config: ContinuousTimeRNNCfg):

        assert len(individual) == self.get_individual_size(input_space, output_size, config)
        optimize_y0 = config.optimize_y0
        delta_t = config.delta_t
        self.optimize_state_boundaries = config.optimize_state_boundaries
        set_principle_diagonal_elements_of_W_negative = config.set_principle_diagonal_elements_of_W_negative
        N_n = config.number_neurons

        input_size = ContinuousTimeRNN._get_size_from_shape(input_space)
        # todo: do the same transformation for output-shapes, too
        # todo: the shape-object contains sometimes min/max values, which should be used to normalize inputs/outputs

        V_size = input_size * N_n
        W_size = N_n * N_n
        T_size = N_n * output_size

        self.delta_t = delta_t
        self.set_principle_diagonal_elements_of_W_negative = set_principle_diagonal_elements_of_W_negative

        # Get weight matrices of current individual
        self.V = np.array([[element] for element in individual[0:V_size]])
        self.W = np.array([[element] for element in individual[V_size:V_size + W_size]])
        self.T = np.array([[element] for element in individual[V_size + W_size:V_size + W_size + T_size]])

        self.V = self.V.reshape([N_n, input_size])
        self.W = self.W.reshape([N_n, N_n])
        self.T = self.T.reshape([N_n, output_size])

        index = V_size + W_size + T_size

        # Initial state values y0
        if optimize_y0:
            self.y0 = np.array([element for element in individual[index:index + N_n]])
            index += N_n
        else:
            self.y0 = np.zeros(N_n)

        self.y = self.y0[:, np.newaxis]

        # Clipping ranges for state boundaries
        if self.optimize_state_boundaries == "per_neuron":
            self.clipping_range_min = np.asarray([-abs(element) for element in individual[index:index + N_n]])
            self.clipping_range_max = np.asarray([abs(element) for element in individual[index + N_n:]])
        elif self.optimize_state_boundaries == "global":
            # apply the same learned state_boundary to all neuron
            raise RuntimeError("not yet implemented")
        elif self.optimize_state_boundaries == "legacy":
            # todo: remove this after implementing and testing global boundaries
            self.clipping_range_min = [-abs(element) for element in individual[index:index + N_n]]
            self.clipping_range_max = [abs(element) for element in individual[index + N_n:]]
        elif self.optimize_state_boundaries == "fixed":
            self.clipping_range_min = np.asarray([-config.clipping_range] * N_n)
            self.clipping_range_max = np.asarray([config.clipping_range] * N_n)

        # Set elements of main diagonal to less than 0
        if set_principle_diagonal_elements_of_W_negative:
            for j in range(N_n):
                self.W[j][j] = -abs(self.W[j][j])

    def step(self, ob: np.ndarray) -> np.ndarray:

        u: np.ndarray = ob[:, np.newaxis]

        # Differential equation
        dydt: np.ndarray = np.dot(self.W, np.tanh(self.y)) + np.dot(self.V, u)

        # Euler forward discretization
        self.y = self.y + self.delta_t * dydt

        # todo remove hack
        if self.optimize_state_boundaries == "legacy":
            for y_min, y_max in zip(self.clipping_range_min, self.clipping_range_max):
                self.y = np.clip(self.y, y_min, y_max)
        else:
            self.y = np.clip(self.y, self.clipping_range_min, self.clipping_range_max)

        # Calculate outputs
        # todo: the output is always between 0 and 1, but for some experiments it should be scaled up to other intervalls
        # gym's output-shape contains all information needed for this todo
        o = np.tanh(np.dot(self.y.T, self.T))

        return o[0]

    @staticmethod
    def _get_size_from_shape(shape: np.shape):
        size = 1
        for val in shape:
            size = size * val
        return size

    @staticmethod
    def get_individual_size(input_space, output_size, config: ContinuousTimeRNNCfg):
        N_n = config.number_neurons

        input_size = ContinuousTimeRNN._get_size_from_shape(input_space)

        individual_size = input_size * N_n + N_n * N_n + N_n * output_size

        if config.optimize_y0:
            individual_size += N_n

        if config.optimize_state_boundaries:
            individual_size += 2 * N_n

        return individual_size
