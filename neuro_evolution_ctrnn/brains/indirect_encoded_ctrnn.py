import numpy as np
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
    red_z: float = 1.3
    green_z: float = 1.6
    blue_z: float = 1.9
    outputs_radius: float = 0.45
    outputs_z: float = -0.5

    input_positions: np.ndarray
    neuron_positions: np.ndarray
    output_positions: np.ndarray

    @classmethod
    def set_class_state(cls, input_positions, neuron_positions, output_positions):
        cls.input_positions = input_positions
        cls.neuron_positions = neuron_positions
        cls.output_positions = output_positions

    def __init__(self, input_space: Space, output_space: Space, individual: np.ndarray, config: IndirectEncodedCtrnnCfg):
        super().__init__(input_space, output_space, individual, config)

        assert len(individual) == self.get_individual_size(config, input_space, output_space)
        self.config = config

        cppn_input_space = Box(-1, 1, (6,))
        cppn_output_space = Box(-1, 1, (1,))

        self.cppn = FeedForwardNumPy(input_space=cppn_input_space, output_space=cppn_output_space,
                                     config=config.ffnn_conf, individual=individual)

        individual_size_ctrnn = ContinuousTimeRNN.get_individual_size(config.ctrnn_conf, input_space, output_space)
        individual_ctrnn = np.zeros(individual_size_ctrnn)

        assert individual_size_ctrnn == len(self.input_positions) * len(self.neuron_positions) + \
               len(self.neuron_positions) * len(self.neuron_positions) + \
               len(self.neuron_positions) * len(self.output_positions)

        self.ctrnn = ContinuousTimeRNN(input_space=input_space, output_space=output_space, config=config.ctrnn_conf,
                                       individual=individual_ctrnn)

        # Convert scipy matrizes to numpy arrays (very hacky to do this here)
        self.ctrnn.V = self.ctrnn.V.toarray()
        self.ctrnn.W = self.ctrnn.W.toarray()
        self.ctrnn.T = self.ctrnn.T.toarray()

        # Decode V using cppn
        for i in range(len(self.neuron_positions)):
            for j in range(len(self.input_positions)):
                cppn_input = np.concatenate((self.input_positions[j], self.neuron_positions[i]))
                self.ctrnn.V[i, j] = self.cppn.step(cppn_input)

        # Decode W using cppn
        for i in range(len(self.neuron_positions)):
            for j in range(len(self.neuron_positions)):
                cppn_input = np.concatenate((self.neuron_positions[j], self.neuron_positions[i]))
                self.ctrnn.W[i, j] = self.cppn.step(cppn_input)

        # Decode T using cppn
        for i in range(len(self.output_positions)):
            for j in range(len(self.neuron_positions)):
                cppn_input = np.concatenate((self.neuron_positions[j], self.output_positions[i]))
                self.ctrnn.T[i, j] = self.cppn.step(cppn_input)

    def brainstep(self, ob: np.ndarray) -> Union[np.ndarray, np.generic]:

        ob = self.get_input_vector_from_observation(ob)

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

        input_positions = cls.get_rgb_positions(input_image_width, input_image_height, cls.red_z, cls.green_z, cls.blue_z)
        neuron_positions = np.random.random((config.ctrnn_conf.number_neurons, 3))
        output_positions = cls.get_circle_positions(output_size, cls.outputs_radius, cls.outputs_z)

        cls.set_class_state(input_positions=input_positions, neuron_positions=neuron_positions, output_positions=output_positions)

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
