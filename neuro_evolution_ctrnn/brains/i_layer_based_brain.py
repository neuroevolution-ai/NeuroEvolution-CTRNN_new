import abc
from brains.i_brain import IBrain, ConfigClass
from gym import Space
import numpy as np
from tools.configurations import ILayerBasedBrainCfg
from typing import List, TypeVar, Generic

LayerdConfigClass = TypeVar('LayerdConfigClass', bound=ILayerBasedBrainCfg)


class ILayerBasedBrain(IBrain, Generic[LayerdConfigClass]):

    @abc.abstractmethod
    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: LayerdConfigClass):
        super().__init__(input_space, output_space, individual, config)
        input_size = self._size_from_space(input_space)
        output_size = self._size_from_space(output_space)
        hidden_layer_structure: List[int] = config.hidden_layer_structure

        # initialize weights out of individual

        individual_index = 0  # progress index
        # initialize emphty
        self.weight_ih = []  # Weights for weighted input values
        self.weight_hh = []  # Weights for weighted stored values
        self.bias_h = []  # Biases
        self.hidden = []  # Initial values for state storage
        self.layer_output = []  # Weights from output last Layer to output nodes
        number_gates = self.get_number_gates()

        # iterate for all given layers in the structure
        for layer in range(len(hidden_layer_structure)):

            # Matrices for weighted input values in calculations
            if layer == 0:
                number_elements = number_gates * hidden_layer_structure[0] * input_size
                self.weight_ih.append(
                    np.array(
                        individual[individual_index: individual_index + number_elements]
                    ).reshape((number_gates, hidden_layer_structure[0], input_size))
                )
                individual_index += number_elements
            else:
                number_elements = number_gates * hidden_layer_structure[layer] * hidden_layer_structure[layer - 1]
                self.weight_ih.append(
                    np.array(
                        individual[individual_index: individual_index + number_elements]
                    ).reshape((number_gates, hidden_layer_structure[layer], hidden_layer_structure[layer - 1]))
                )
                individual_index += number_elements

            # Matrices for weighted state values in calculations
            if config.diagonal_hidden_to_hidden:  # Whether each neuron can only access its own state
                self.weight_hh.append(
                    [np.diag(individual[
                             individual_index + k * hidden_layer_structure[layer] * hidden_layer_structure[layer]:
                             individual_index + k * hidden_layer_structure[layer] * hidden_layer_structure[layer] +
                             hidden_layer_structure[layer]])
                     for k in range(number_gates)
                     ]
                )
                individual_index += number_gates * hidden_layer_structure[layer]
            else:
                number_elements = number_gates * hidden_layer_structure[layer] * hidden_layer_structure[layer]
                self.weight_hh.append(
                    np.array(
                        individual[individual_index: individual_index + number_elements]
                    ).reshape((number_gates, hidden_layer_structure[layer], hidden_layer_structure[layer]))
                )
                individual_index += number_elements

            # initialize biases

            # Biases for gates
            if config.use_bias:
                number_elements = hidden_layer_structure[layer] * number_gates
                self.bias_h.append(
                    np.array(
                        individual[individual_index: individual_index + number_elements]
                    ).reshape((number_gates, hidden_layer_structure[layer]))
                )
                individual_index += number_elements
            else:
                self.bias_h.append(np.zeros((number_gates, hidden_layer_structure[layer])).astype(np.float32))

            # initialize initial state values

            if config.optimize_initial_neuron_state:
                number_elements = hidden_layer_structure[layer]
                self.hidden.append(np.array(
                    individual[individual_index: individual_index + number_elements]
                ))
                individual_index += number_elements
            else:
                self.hidden.append(np.zeros((hidden_layer_structure[layer])))
            self.layer_output.append(np.zeros((hidden_layer_structure[layer])))
        # for end

        # Matrix for transforming output of last layer into output neurons
        number_elements = hidden_layer_structure[len(hidden_layer_structure) - 1] * output_size
        self.weight_ho = np.array(
            individual[individual_index: individual_index + number_elements]
        ).reshape((output_size, hidden_layer_structure[len(hidden_layer_structure) - 1]))
        individual_index += number_elements

        # Has all values been used and therefore does get_individual_size() provide the right number?
        assert individual_index == len(individual)

    @staticmethod
    @abc.abstractmethod
    def get_number_gates():
        # How many Gates are used in the specific network?
        # Haw many matrices are needed for each layer to calculate the next state and output value
        pass

    @classmethod
    def get_individual_size(cls, config: ILayerBasedBrainCfg, input_space: Space, output_space: Space):

        number_gates = cls.get_number_gates()
        input_size = cls._size_from_space(input_space)
        output_size = cls._size_from_space(output_space)
        hidden_struc = config.hidden_layer_structure

        individual_size = 0

        for layer in range(len(hidden_struc)):
            # Matrices for weighted input values
            if layer == 0:  # The first Layer don't has an output from the previous layer, but the input values
                individual_size += number_gates * input_size * hidden_struc[0]
            else:
                individual_size += number_gates * hidden_struc[layer] * hidden_struc[layer - 1]

            # Matrices for weighted state values
            if config.diagonal_hidden_to_hidden:
                individual_size += number_gates * hidden_struc[layer]
            else:
                individual_size += number_gates * hidden_struc[layer] * hidden_struc[layer]

            # initialize biases
            if config.use_bias:
                individual_size += hidden_struc[layer] * number_gates

            # Hidden values
            if config.optimize_initial_neuron_state:
                individual_size += hidden_struc[layer]
        # for end

        # Matrix for transforming output of last layer into output neurons
        individual_size += hidden_struc[len(hidden_struc) - 1] * output_size
        return individual_size

    def brainstep(self, ob: np.ndarray):

        layer_input = ob
        # iterate for all given layers
        for layer in range(len(self.hidden)):
            if layer == 0:
                x = layer_input
            else:
                x = self.layer_output[layer - 1]

            # Returns a list with two elements.
            # The first element is the calculated new hidden cell state, the second is the layer output
            # Necessary for LSTM
            layer_result = self.layer_step(x, self.weight_ih[layer], self.weight_hh[layer], self.bias_h[layer],
                                           self.hidden[layer])
            self.hidden[layer] = layer_result[0]
            self.layer_output[layer] = layer_result[1]
        return np.dot(self.weight_ho, self.layer_output[len(self.layer_output) - 1].T)

    @staticmethod
    @abc.abstractmethod
    def layer_step(layer_input: np.ndarray, weight_ih, weight_hh, bias_h, hidden):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
