import numpy as np
from gym import Space

from brains.brain import register_brain
from recurrent.recurrent_based_brain import RecurrentBasedBrain
from tools.configurations import ElmanCfg


@register_brain("Elman", ElmanCfg)
class Elman(RecurrentBasedBrain[ElmanCfg]):

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: ElmanCfg):
        super().__init__(input_space, output_space, individual, config)

    @staticmethod
    def get_number_hidden_values():
        return 1

    @staticmethod
    def get_number_gates():
        return 1

    @staticmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        result = np.tanh(
            np.dot(weight_ih[0], layer_input) +
            np.dot(weight_hh[0], hidden[0]) +
            bias_h[0])
        return [[result], result]

    def get_neuron_states(self):
        return np.array([item for sublist in self.hidden for item in sublist])

    def get_internal_weight_matrix(self):
        # A complete matrix for the weighted input values
        # Is used to draw the lines in the visualization
        result: np.array = self.weight_hh[0][0]
        for i in range(1, len(self.hidden)):
            result = self.append_matrix_horizontally(result, self.append_matrix_vertically(
                np.zeros((len(self.weight_ih[i][0]), len(result[0]) - len(self.weight_ih[i][0][0]))).tolist(),
                self.weight_ih[i][0]))

            result = self.append_matrix_vertically(result, self.append_matrix_horizontally(
                np.zeros((len(result) - len(self.weight_hh[i][0]), len(self.weight_hh[i][0][0]))).tolist(),
                self.weight_hh[i][0].tolist()))
        return np.array(result)

    @staticmethod
    def append_matrix_vertically(matrix1, matrix2):
        return np.concatenate((matrix1, matrix2), 1)

    @staticmethod
    def append_matrix_horizontally(matrix1, matrix2):
        return np.concatenate((matrix1, matrix2), 0)

    def get_input_weight_matrix(self):
        # TODO visualize bias
        return self.append_matrix_horizontally(
            self.weight_ih[0][0],
            np.zeros((len(self.get_neuron_states()) - len(self.weight_ih[0][0]), len(self.weight_ih[0][0][0])))
        )

    def get_output_weight_matrix(self):
        return self.append_matrix_vertically(
            np.zeros((len(self.weight_ho), len(self.get_neuron_states()) - len(self.weight_ho[0]))),
            self.weight_ho
        )
