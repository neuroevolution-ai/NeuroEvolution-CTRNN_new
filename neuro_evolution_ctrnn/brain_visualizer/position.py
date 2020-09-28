import networkx as nx
import numpy as np

from brain_visualizer import brain_visualizer


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

        pos = nx.spring_layout(g, k=1, weight="myweight", iterations=50, scale=visualizer.h / 2 - 100)

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

    # Calculate Input or Output Positions based on number of Neurons and radius of Neurons
    @staticmethod
    def get_input_output_positions(visualizer: "brain_visualizer.BrainVisualizer", number_neurons: int,
                                   is_input: bool) -> dict:
        positions_dict = {}
        if is_input:
            x = ((1 * visualizer.w) / 12)
            x2 = ((1 * visualizer.w) / 18)
            x3 = ((2 * visualizer.w) / 18)
            radius = visualizer.input_neuron_radius
        else:
            x = ((11 * visualizer.w) / 12)
            x2 = ((16 * visualizer.w) / 18)
            x3 = ((17 * visualizer.w) / 18)
            radius = visualizer.neuron_radius

        # Place Neurons in one row if there is enough place, else take two rows
        for i in range(number_neurons):
            if ((visualizer.h - 100) / (number_neurons * radius * 2)) > 1:
                x_pos = x
                y_pos = (((radius * 2) + (visualizer.h / 2)) - (
                        number_neurons * radius) + (i * radius * 2))
                positions_dict[i] = [x_pos, y_pos]
            else:
                if i % 2:
                    x_pos = x2
                    y_pos = ((radius * 2) + (visualizer.h / 2)) - (
                            (number_neurons * radius) / 2) + (
                                    i * radius)
                    positions_dict[i] = [x_pos, y_pos]
                else:
                    x_pos = x3
                    y_pos = ((radius * 2) + (visualizer.h / 2)) - (
                            (number_neurons * radius) / 2) + (
                                    i * radius)
                    positions_dict[i] = [x_pos, y_pos]
        return positions_dict
