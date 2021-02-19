import numpy as np
from scipy import sparse
from typing import Union
from gym.spaces import Space, Box, flatdim
from brains import IBrain, register_brain
from brains.continuous_time_rnn import ContinuousTimeRNN
from brains.ffnn import FeedForwardNumPy
from tools.configurations import IndirectEncodedCtrnnCfg


# noinspection PyPep8Naming
@register_brain("Indirect_CTRNN", IndirectEncodedCtrnnCfg)
class IndirectEncodedCtrnn(IBrain[IndirectEncodedCtrnnCfg]):

    # Parameters TODO: Put them to anywhere else
    red_z: float
    green_z: float
    blue_z: float
    outputs_radius: float
    outputs_z: float

    input_positions: np.ndarray
    neuron_positions: np.ndarray
    output_positions: np.ndarray
    cppn_inputs_V: np.ndarray
    cppn_inputs_W: np.ndarray
    cppn_inputs_T: np.ndarray

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: IndirectEncodedCtrnnCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        self.config = config

        cppn_input_space = Box(-1, 1, (6,))
        cppn_output_space = Box(-1, 1, (1,))

        self.cppn = FeedForwardNumPy(input_space=cppn_input_space, output_space=cppn_output_space, config=config.ffnn_conf, individual=individual)

        individual_ctrnn_V = self.cppn.predict(self.cppn_inputs_V)
        individual_ctrnn_W = self.cppn.predict(self.cppn_inputs_W)
        individual_ctrnn_T = self.cppn.predict(self.cppn_inputs_T)

        #individual_ctrnn_V = np.random.random(flatdim(input_space) * config.ctrnn_conf.number_neurons)
        #individual_ctrnn_W = np.random.random(config.ctrnn_conf.number_neurons* config.ctrnn_conf.number_neurons)
        #individual_ctrnn_T = np.random.random(flatdim(output_space) * config.ctrnn_conf.number_neurons)

        individual_ctrnn_V = individual_ctrnn_V.flatten()# * (np.random.rand(len(individual_ctrnn_V)) < 0.05)
        individual_ctrnn_W = individual_ctrnn_W.flatten()
        individual_ctrnn_T = individual_ctrnn_T.flatten()

        individual_ctrnn = np.concatenate((individual_ctrnn_V, individual_ctrnn_W, individual_ctrnn_T))

        self.ctrnn = ContinuousTimeRNN(input_space=input_space, output_space=output_space, config=config.ctrnn_conf,
                                       individual=individual_ctrnn)

        # Convert scipy matrizes (very hacky to do this here)
        self.ctrnn.V = sparse.csr_matrix(self.ctrnn.V.toarray(), dtype=float)
        self.ctrnn.W = sparse.csr_matrix(self.ctrnn.W.toarray(), dtype=float)
        self.ctrnn.T = sparse.csr_matrix(self.ctrnn.T.toarray(), dtype=float)

    def brainstep(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:

        assert ob.ndim == 3
        assert ob.shape == self.ctrnn.input_space.shape

        ob = self.get_input_vector_from_observation(ob)

        assert ob.ndim == 1

        return self.ctrnn.brainstep(ob=ob)


    @classmethod
    def get_free_parameter_usage(cls, config: IndirectEncodedCtrnnCfg, input_space: Space, output_space: Space):

        cppn_input_space = Box(-1, 1, (6,))
        cppn_output_space = Box(-1, 1, (1,))

        individual_size = FeedForwardNumPy.get_individual_size(config.ffnn_conf, cppn_input_space, cppn_output_space)

        return {'individual_size': individual_size}

    @classmethod
    def generate_and_set_class_state(cls, config: IndirectEncodedCtrnnCfg, input_space: Space, output_space: Space):
        ContinuousTimeRNN.generate_and_set_class_state(config=config.ctrnn_conf, input_space=input_space, output_space=output_space)

        input_image_width = input_space.shape[0]
        input_image_height = input_space.shape[1]
        output_size = flatdim(output_space)

        # TODO: Put them to anywhere else
        cls.red_z = 1.3
        cls.green_z = 1.6
        cls.blue_z = 1.9
        cls.outputs_radius = 0.45
        cls.outputs_z = -0.5

        cls.input_positions = cls.get_rgb_positions(input_image_width, input_image_height, cls.red_z, cls.green_z, cls.blue_z)
        cls.neuron_positions = np.random.random((config.ctrnn_conf.number_neurons, 3))
        cls.output_positions = cls.get_circle_positions(output_size, cls.outputs_radius, cls.outputs_z)

        individual_size_ctrnn = ContinuousTimeRNN.get_individual_size(config.ctrnn_conf, input_space, output_space)

        assert individual_size_ctrnn == len(cls.input_positions) * len(cls.neuron_positions) + len(
            cls.neuron_positions) * len(cls.neuron_positions) + len(cls.neuron_positions) * len(cls.output_positions)

        cls.cppn_inputs_V = np.zeros((len(cls.input_positions) * len(cls.neuron_positions), 6))
        cls.cppn_inputs_W = np.zeros((len(cls.neuron_positions) * len(cls.neuron_positions), 6))
        cls.cppn_inputs_T = np.zeros((len(cls.output_positions) * len(cls.neuron_positions), 6))

        # Decode V using cppn
        index_V = 0
        for i in range(len(cls.neuron_positions)):
            for j in range(len(cls.input_positions)):
                cls.cppn_inputs_V[index_V] = np.concatenate((cls.input_positions[j], cls.neuron_positions[i]))
                index_V += 1

        # Decode W using cppn
        index_W = 0
        for i in range(len(cls.neuron_positions)):
            for j in range(len(cls.neuron_positions)):
                cls.cppn_inputs_W[index_W] = np.concatenate((cls.neuron_positions[j], cls.neuron_positions[i]))
                index_W += 1

        # Decode T using cppn
        index_T = 0
        for i in range(len(cls.output_positions)):
            for j in range(len(cls.neuron_positions)):
                cls.cppn_inputs_T[index_T] = np.concatenate((cls.neuron_positions[j], cls.output_positions[i]))
                index_T += 1

        assert index_V + index_W + index_T == individual_size_ctrnn

        cls.cppn_inputs_V = cls.cppn_inputs_V.reshape(-1, 6, 1)
        cls.cppn_inputs_W = cls.cppn_inputs_W.reshape(-1, 6, 1)
        cls.cppn_inputs_T = cls.cppn_inputs_T.reshape(-1, 6, 1)

    @staticmethod
    def get_rgb_positions(width: int, height: int,
                          red_position_z: float, green_position_z: float, blue_position_z: float) -> np.ndarray:
        space_x = np.linspace(0, 1, width)
        space_y = np.linspace(0, 1, height)

        grid_x, grid_y = np.meshgrid(space_x, space_y)
        rgb_positions_x = grid_x.flatten()
        rgb_positions_y = grid_y.flatten()

        red_positions = np.column_stack(
            (rgb_positions_x, rgb_positions_y, np.ones(width * height) * red_position_z))
        green_positions = np.column_stack(
            (rgb_positions_x, rgb_positions_y, np.ones(width * height) * green_position_z))
        blue_positions = np.column_stack(
            (rgb_positions_x, rgb_positions_y, np.ones(width * height) * blue_position_z))

        return np.vstack((red_positions, green_positions, blue_positions))

    @staticmethod
    def get_circle_positions(number_positions: int, radius: float, position_z: float) -> np.ndarray:
        circle_positions_x = np.zeros(number_positions)
        circle_positions_y = np.zeros(number_positions)

        angles = np.linspace(0, 2 * np.pi, number_positions, endpoint=False)

        for i in range(number_positions):
            circle_positions_x[i] = radius * np.cos(angles[i]) + 0.5
            circle_positions_y[i] = radius * np.sin(angles[i]) + 0.5

        circle_positions_z = np.ones(number_positions) * position_z

        return np.column_stack((circle_positions_x, circle_positions_y, circle_positions_z))

    @staticmethod
    def get_input_vector_from_observation(observation: np.ndarray) -> np.ndarray:
        observation_scaled = observation / 255.0

        observation_red = observation_scaled[:, :, 0].flatten()
        observation_green = observation_scaled[:, :, 1].flatten()
        observation_blue = observation_scaled[:, :, 2].flatten()

        return np.concatenate((observation_red, observation_green, observation_blue))
