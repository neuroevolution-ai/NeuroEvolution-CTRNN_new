import math
from typing import Tuple

import networkx as nx
import numpy as np

from brain_visualizer import brain_visualizer
from tools.configurations import IBrainCfg


class Positions:

    @staticmethod
    def get_graph_positions(visualizer: "brain_visualizer.BrainVisualizer") -> dict:
        brain_state = visualizer.brain.y
        brain_weight = visualizer.brain.W.todense()

        # Create Graph by adding Nodes and Edges separately
        g = nx.Graph(brain="CTRNN")

        for i in range(len(brain_state)):
            g.add_node(i)

        for index, value in np.ndenumerate(brain_weight):
            g.add_edges_from([(index[0], index[1], {'myweight': value})])

        pos = nx.spring_layout(g, k=1.5, weight="myweight", iterations=50, scale=visualizer.h / 2 - 100)

        # Adapt positions from spring-layout Method to pygame windows
        graph_positions_dict = {}
        for each in pos:
            position = pos[each]
            pos_x = int(position[0] + (visualizer.w / 2))
            if pos_x > (visualizer.w / 2):
                pos_x = pos_x + 50
            if pos_x < (visualizer.w / 2):
                pos_x = pos_x - 50
            pos_y = int(position[1] + (visualizer.h / 2)) + 60
            graph_positions_dict[each] = [pos_x, pos_y]

        return graph_positions_dict

    @staticmethod
    def calculate_input_positions(visualizer: "brain_visualizer.BrainVisualizer", in_values: np.ndarray):
        # Height and width of the "Input Box" which is the part of the window which contains the input neurons, below
        # the info box
        input_box_height = visualizer.h - visualizer.info_box_size
        input_box_width = visualizer.input_box_width

        if len(in_values.shape) == 1:
            # Leave space on the borders
            space = 25
            input_box_height -= 2 * space
            input_box_width -= 2 * space

            size_fits = False
            neurons_per_column = -1
            while not size_fits:
                neurons_per_column = math.floor(input_box_height / (2 * visualizer.input_neuron_radius))
                columns = math.ceil(in_values.size / neurons_per_column)
                input_neuron_width = columns * visualizer.input_neuron_radius * 2

                if input_neuron_width > input_box_width:
                    visualizer.input_neuron_radius -= 5

                    if visualizer.input_neuron_radius <= 0:
                        visualizer.input_neuron_radius = 1
                        break
                else:
                    size_fits = True

            if neurons_per_column <= 0:
                # Hacky way to avoid an error
                neurons_per_column = 5

            # Add another spacer, so that the input_box is centered with respect to the height
            adjusted_height_space = round(
                (input_box_height - neurons_per_column * visualizer.input_neuron_radius * 2) / 2)

            current_x = space
            current_y = visualizer.info_box_size + space + adjusted_height_space

            positions_dict = {}
            current_neurons_in_column = 0

            for i in range(len(in_values)):
                positions_dict[i] = [current_x, current_y]
                current_neurons_in_column += 1
                current_y += visualizer.input_neuron_radius * 2

                # If a column is full reset the y-value and shift the x-value by one diameter of a neuron
                if current_neurons_in_column > neurons_per_column:
                    current_neurons_in_column = 0
                    current_x += visualizer.input_neuron_radius * 2
                    current_y = visualizer.info_box_size + space + adjusted_height_space

            return positions_dict

        elif len(in_values.shape) == 3:
            rows_per_block, columns_per_block, blocks = in_values.shape

            # Use empty space on top and bottom and left and right end of the boxes respectively
            space = 25

            block_height = int(input_box_height / 3.0) - space
            block_width = input_box_width - space

            visualizer.input_neuron_radius = min(block_height / rows_per_block, block_width / columns_per_block)
            visualizer.input_neuron_radius = round(visualizer.input_neuron_radius / 2)

            if visualizer.input_neuron_radius == 0:
                raise RuntimeError("""Too many input values provided. They cannot be drawn, please consider increasing
                 the windows size or decreasing the number of input values.""")

            positions_dict = {}
            current_x = 0
            current_y = 0
            index = 0
            # Iterate through the blocks, x value is always the same, y value needs to be adjusted accordingly
            for i in range(blocks):
                current_x = int(space / 2.0)
                current_y = visualizer.info_box_size + int(space / 2.0) + i * (block_height + space)
                # Draw the rows, therefore increase the current y value by the neuron diameter and reset the x value
                for x in range(rows_per_block):
                    # Draw the columns, therefore increase the x value by the neuron diameter
                    for y in range(columns_per_block):
                        positions_dict[index] = [current_x, current_y]
                        index += 1
                        current_x += visualizer.input_neuron_radius * 2
                    current_x = int(space / 2.0)
                    current_y += visualizer.input_neuron_radius * 2

            # Simply add one neuron to the last block if a bias is used
            if visualizer.brain_config.use_bias:
                positions_dict[index] = [current_x, current_y]

            return positions_dict
        else:
            # Only one dimensional or three dimensional input is allowed
            raise RuntimeError("Only one-dimensional or three-dimensional input is supported for the BrainVisualizer.")

    # Calculate Input or Output Positions based on number of Neurons and radius of Neurons
    @staticmethod
    def get_input_output_positions(visualizer: "brain_visualizer.BrainVisualizer", number_neurons: int,
                                   is_input: bool, rgb_shape: Tuple[int, int, int] = None) -> dict:
        # Sum of space between the info box and bottom of the window and the neurons
        # extra_space = 20

        radius = visualizer.input_neuron_radius if is_input else visualizer.neuron_radius

        # columns_needed: int = math.ceil(
        #     (number_neurons * radius * 2) / (visualizer.h - visualizer.info_box_size - extra_space)
        # )

        # neurons_per_column = math.ceil(number_neurons / columns_needed)

        if is_input:
            # x = extra_space
            x = ((1 * visualizer.w) / 12)
            x2 = ((1 * visualizer.w) / 18)
            x3 = ((2 * visualizer.w) / 18)
        else:
            # Calculate offset from right window border
            # x = visualizer.w - extra_space - (columns_needed * radius * 2)
            x = ((11 * visualizer.w) / 12)
            x2 = ((16 * visualizer.w) / 18)
            x3 = ((17 * visualizer.w) / 18)

        # current_y = visualizer.info_box_size + extra_space
        # j = 0
        positions_dict = {}

        if rgb_shape is not None:
            current_y = visualizer.info_box_size + 20
            index = 0
            x, y, rgb = rgb_shape
            for _ in range(rgb):
                current_x = 20

                for i in range(x):
                    for j in range(y):
                        positions_dict[index] = [current_x, current_y]
                        index += 1
                        current_x += radius * 2 + 5
                    current_x = 20
                    current_y += radius * 2 + 5
                current_y += 90

            positions_dict[index] = [current_x, current_y]
            return positions_dict

            # for i in range(number_neurons):
            #     np.array_equal(in_values[:, :, 0].flatten(), in_values.flatten()[0::3])

        # Place Neurons in one row if there is enough place, else take two rows
        for i in range(number_neurons):
            # positions_dict[i] = [x, current_y]
            # current_y += radius * 2
            # j += 1

            # if j > neurons_per_column:
            #     current_y = visualizer.info_box_size + extra_space
            #     j = 0
            #     x += radius * 2

            if ((visualizer.h - 100) / (number_neurons * radius * 2)) > 1:
                x_pos = x
                y_pos = (((radius * 2) + (visualizer.h / 2)) - (
                        number_neurons * radius) + (i * radius * 2))
                positions_dict[i] = [x_pos, y_pos]
            else:
                if i % 2:
                    x_pos = x2
                    y_pos = ((radius * 2) + (visualizer.h / 2)) - ((number_neurons * radius) / 2) + (i * radius)
                    positions_dict[i] = [x_pos, y_pos]
                else:
                    x_pos = x3
                    y_pos = ((radius * 2) + (visualizer.h / 2)) - (
                            (number_neurons * radius) / 2) + (
                                    i * radius)
                    positions_dict[i] = [x_pos, y_pos]
        return positions_dict
