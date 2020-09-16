import networkx as nx
import numpy as np


class Positions:

    @staticmethod
    def get_graph_positions(brain_visualizer):
        brain_state = brain_visualizer.brain.y
        brain_weight = brain_visualizer.brain.W.todense()

        # Create Graph by adding Nodes and Edges separately
        g = nx.Graph(brain="CTRNN")

        for i in range(len(brain_state)):
            g.add_node(i)

        for index, value in np.ndenumerate(brain_weight):
            g.add_edges_from([(index[0], index[1], {'myweight': value})])

        pos = nx.spring_layout(g, k=1, weight="myweight", iterations=50, scale=brain_visualizer.h / 2 - 100)

        # Adapt positions from spring-layout Method to pygame windows
        graph_positions_dict = {}
        for each in pos:
            position = pos[each]
            pos_x = int(position[0] + (brain_visualizer.w / 2))
            if pos_x > (brain_visualizer.w / 2):
                pos_x = pos_x + 50
            if pos_x < (brain_visualizer.w / 2):
                pos_x = pos_x - 50
            pos_y = int(position[1] + (brain_visualizer.h / 2)) + 60
            graph_positions_dict[each] = [pos_x, pos_y]

        return graph_positions_dict

    # Calculate Input or Output Positions based on number of Neurons and radius of Neurons
    @staticmethod
    def get_input_output_positions(brain_visualizer, number_neurons: int, is_input: bool):
        positions_dict = {}
        if is_input:
            x = ((1 * brain_visualizer.w) / 12)
            x2 = ((1 * brain_visualizer.w) / 18)
            x3 = ((2 * brain_visualizer.w) / 18)
        else:
            x = ((11 * brain_visualizer.w) / 12)
            x2 = ((16 * brain_visualizer.w) / 18)
            x3 = ((17 * brain_visualizer.w) / 18)

        # Place Neurons in one row if there is enough place, else take two rows
        for i in range(number_neurons):
            if ((brain_visualizer.h - 100) / (number_neurons * brain_visualizer.neuron_radius * 2)) > 1:
                x_pos = x
                y_pos = (((brain_visualizer.neuron_radius * 2) + (brain_visualizer.h / 2)) - (
                            number_neurons * brain_visualizer.neuron_radius) + (i * brain_visualizer.neuron_radius * 2))
                positions_dict[i] = [x_pos, y_pos]
            else:
                if i % 2:
                    x_pos = x2
                    y_pos = ((brain_visualizer.neuron_radius * 2) + (brain_visualizer.h / 2)) - (
                            (number_neurons * brain_visualizer.neuron_radius) / 2) + (
                                    i * brain_visualizer.neuron_radius)
                    positions_dict[i] = [x_pos, y_pos]
                else:
                    x_pos = x3
                    y_pos = ((brain_visualizer.neuron_radius * 2) + (brain_visualizer.h / 2)) - (
                            (number_neurons * brain_visualizer.neuron_radius) / 2) + (
                                    i * brain_visualizer.neuron_radius)
                    positions_dict[i] = [x_pos, y_pos]
        return positions_dict
