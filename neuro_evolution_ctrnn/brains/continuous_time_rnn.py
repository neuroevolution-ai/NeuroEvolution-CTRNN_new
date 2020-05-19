from typing import List

import numpy as np
from collections import namedtuple

ContrinuousTimeRNNCfg = namedtuple("ContrinuousTimeRNNCfg", [
    "optimize_y0",
    "delta_t",
    "optimize_state_boundaries",
    "set_principle_diagonal_elements_of_W_negative",
    "number_neurons",
    "clipping_range_min", "clipping_range_max"
])


class ContinuousTimeRNN:

    def __init__(self, input_size: int, output_size: int, individual: List[float], config: ContrinuousTimeRNNCfg):

        optimize_y0 = config.optimize_y0
        delta_t = config.delta_t
        optimize_state_boundaries = config.optimize_state_boundaries
        set_principle_diagonal_elements_of_W_negative = config.set_principle_diagonal_elements_of_W_negative
        N_n = config.number_neurons

        V_size = input_size * N_n
        W_size = N_n * N_n
        T_size = N_n * output_size

        self.delta_t = delta_t
        self.set_principle_diagonal_elements_of_W_negative = set_principle_diagonal_elements_of_W_negative

        # Get weight matrizes of current individual
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
        if optimize_state_boundaries:
            self.clipping_range_min = np.asarray([-abs(element) for element in individual[index:index + N_n]])
            self.clipping_range_max = np.asarray([abs(element) for element in individual[index + N_n:]])
        else:
            self.clipping_range_min = np.asarray([-config.clipping_range] * N_n)
            self.clipping_range_max = np.asarray([config.clipping_range] * N_n)

        self.clipping_range_min = self.clipping_range_min[:, np.newaxis]
        self.clipping_range_max = self.clipping_range_max[:, np.newaxis]

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

        # Clip y to state boundaries
        self.y = np.clip(self.y, self.clipping_range_min, self.clipping_range_max)

        # Calculate outputs
        o = np.tanh(np.dot(self.y.T, self.T))

        return o[0]

    @staticmethod
    def get_individual_size(input_size, output_size, config: ContrinuousTimeRNNCfg):
        N_n = config.number_neurons

        individual_size = input_size * N_n + N_n * N_n + N_n * output_size

        if config.optimize_y0:
            individual_size += N_n

        if config.optimize_state_boundaries:
            individual_size += 2 * N_n

        return individual_size
