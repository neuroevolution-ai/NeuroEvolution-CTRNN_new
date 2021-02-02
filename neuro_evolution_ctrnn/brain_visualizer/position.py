import math

import networkx as nx
import numpy as np

from brain_visualizer import brain_visualizer


class Positions:

    @staticmethod
    def get_graph_positions(visualizer: "brain_visualizer.BrainVisualizer") -> dict:
        brain_state = visualizer.brain.get_neuron_states()
        brain_weight = visualizer.brain.get_internal_weight_matrix()

        # Create Graph by adding Nodes and Edges separately
        g = nx.Graph(brain="CTRNN")

        for i in range(len(brain_state)):
            g.add_node(i)

        for index, value in np.ndenumerate(brain_weight):
            g.add_edges_from([(index[0], index[1], {'edge_weight': value})])

        pos = nx.spring_layout(g, k=1.5, weight="edge_weight", scale=visualizer.h / 2 - 100, seed=0)

        # Adapt positions from spring-layout Method to pygame windows
        graph_positions_dict = {}
        for each in pos:
            position = pos[each]
            pos_x = int(position[0] + (visualizer.w / 2))
            pos_y = int(position[1] + (visualizer.h / 2)) + visualizer.info_box_height
            graph_positions_dict[each] = [pos_x, pos_y]

        return graph_positions_dict

    @staticmethod
    def get_columns_one_dimensional(neuron_radius: int, number_of_neurons: int, box_height: int, box_width: int):
        size_fits = False
        neurons_per_column = -1
        columns = -1
        adjusted_neuron_radius = neuron_radius
        while not size_fits:
            neurons_per_column = math.floor(box_height / (2 * adjusted_neuron_radius))
            columns = math.ceil(number_of_neurons / neurons_per_column)
            input_neuron_width = columns * adjusted_neuron_radius * 2

            if input_neuron_width > box_width:
                adjusted_neuron_radius -= 5

                if adjusted_neuron_radius <= 0:
                    adjusted_neuron_radius = 1
                    break
            else:
                size_fits = True

        # TODO this is a little bit hacky
        if neurons_per_column <= 0 or columns <= 0:
            # Hacky way to avoid an error
            neurons_per_column = 5
            columns = 1

        return neurons_per_column, columns, adjusted_neuron_radius

    @staticmethod
    def calculate_positions(visualizer: "brain_visualizer.BrainVisualizer", values: np.ndarray, is_input: bool = False):
        # Height and width of the "Input Box" which is the part of the window which contains the input neurons, below
        # the info box
        box_height = visualizer.h - visualizer.info_box_height
        box_width = visualizer.input_box_width

        positions_dict = {}

        if is_input and visualizer.rgb_input:
            rows_per_block, columns_per_block, blocks = visualizer.input_shape

            # Use empty space on top and bottom and left and right end of the boxes respectively
            space = 25

            block_height = int(box_height / 3.0) - space
            block_width = box_width - space

            visualizer.input_neuron_radius = min(block_height / rows_per_block, block_width / columns_per_block)
            visualizer.input_neuron_radius = round(visualizer.input_neuron_radius / 2)

            if visualizer.input_neuron_radius == 0:
                raise RuntimeError("""Too many input values provided. They cannot be drawn, please consider increasing
                 the window size or decreasing the number of input values.""")

            current_x = 0
            current_y = 0
            index = 0

            # Iterate through the blocks, x value is always the same, y value needs to be adjusted accordingly
            for i in range(blocks):
                current_x = int(space / 2.0)
                current_y = visualizer.info_box_height + int(space / 2.0) + i * (block_height + space)
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
        else:
            # Leave space on the borders
            space = 25
            box_height -= 2 * space
            box_width -= 2 * space

            if is_input:
                neurons_per_column, columns, adjusted_neuron_radius = Positions.get_columns_one_dimensional(
                    visualizer.input_neuron_radius, values.size, box_height, box_width)
                visualizer.input_neuron_radius = adjusted_neuron_radius

                current_x = space

            else:
                neurons_per_column, columns, adjusted_neuron_radius = Positions.get_columns_one_dimensional(
                    visualizer.output_neuron_radius, values.size, box_height, box_width)
                visualizer.output_neuron_radius = adjusted_neuron_radius

                current_x = visualizer.w - space - (columns * adjusted_neuron_radius * 2)

            # Add another spacer, so that the input_box is centered with respect to the height
            # Take the min because for small outputs it can occur that not even one column is full
            adjusted_height_space = round(
                (box_height - min(neurons_per_column, values.size) * adjusted_neuron_radius * 2) / 2)

            default_y = current_y = visualizer.info_box_height + space + adjusted_height_space

            current_neurons_in_column = 0

            for i in range(len(values)):
                positions_dict[i] = [current_x, current_y]
                current_neurons_in_column += 1
                current_y += adjusted_neuron_radius * 2

                # If a column is full reset the y-value and shift the x-value by one diameter of a neuron
                if current_neurons_in_column > neurons_per_column:
                    current_neurons_in_column = 0
                    current_x += adjusted_neuron_radius * 2
                    current_y = default_y

        return positions_dict
