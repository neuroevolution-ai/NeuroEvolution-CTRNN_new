import pygame
import numpy as np
import math
from typing import Tuple

from brain_visualizer import brain_visualizer


class Weights:

    @staticmethod
    def arrow(screen: pygame.Surface, color: Tuple[int, int, int], tricolor: Tuple[int, int, int],
              start: Tuple[int, int], end: Tuple[int, int], trirad: int, width: int) -> None:
        if width >= 1:
            pygame.draw.line(screen, color, start, end, width)
            rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
            pygame.draw.polygon(screen, tricolor, (
                (
                    end[0] + trirad * math.sin(math.radians(rotation)),
                    end[1] + trirad * math.cos(math.radians(rotation))),
                (
                    end[0] + trirad * math.sin(math.radians(rotation - 120)),
                    end[1] + trirad * math.cos(math.radians(rotation - 120))), (
                    end[0] + trirad * math.sin(math.radians(rotation + 120)),
                    end[1] + trirad * math.cos(math.radians(rotation + 120)))))

    @staticmethod
    def draw_connection(visualizer: "brain_visualizer.BrainVisualizer", start_pos, end_pos, weight,
                        is_input: bool = False):
        if weight > 0.0:
            weight_color = visualizer.color_positive_weight
        else:
            weight_color = visualizer.color_negative_weight

        if is_input:
            if weight > 0.0:
                weight_color = visualizer.color_input_connections_positive
            else:
                weight_color = visualizer.color_input_connections_negative

        width = int(abs(weight)) + visualizer.weight_val

        if visualizer.weight_val == 0 and width < 1:
            width = 1

        if visualizer.weights_direction:
            # Angle of the line between both points to the x-axis
            rotation = math.atan2((end_pos[1] - start_pos[1]), (end_pos[0] - start_pos[0]))

            # Point, angle and length of the line for the endpoint of the arrow
            trirad = 5 + width
            arrow_length = (-1 * (visualizer.neuron_radius + trirad + 5))
            arrow_end = (end_pos[0] + arrow_length * math.cos(rotation),
                         end_pos[1] + arrow_length * math.sin(rotation))

            if rotation != 0:
                Weights.arrow(visualizer.screen, weight_color, weight_color, start_pos, arrow_end,
                              trirad, width)
        else:
            pygame.draw.line(visualizer.screen, weight_color, (int(start_pos[0]), int(start_pos[1])),
                             (int(end_pos[0]), int(end_pos[1])), width)

    @staticmethod
    def draw_maximum_weights(visualizer: "brain_visualizer.BrainVisualizer", start_pos_dict: dict, end_pos_dict: dict,
                             weight_matrix, _input: np.ndarray = None) -> None:
        if visualizer.rgb_input:
            # If RGB input is present the weight matrix needs to be reordered a bit. In the brain the RGB values get
            # flattened, therefore three consecutive values in the weight_matrix form one pixel (red, green and blue).
            # The input although is ordered in red, green and blue blocks separately in the BrainVisualizer. So
            # simply concatenate first the red values, then the green values and then the blue values
            if visualizer.brain_config.use_bias:
                bias = weight_matrix[-1]
                weight_matrix = weight_matrix[:-1]

            weight_matrix = np.concatenate((weight_matrix[::3], weight_matrix[1::3], weight_matrix[2::3]))

            if visualizer.brain_config.use_bias:
                # noinspection PyUnboundLocalVariable
                weight_matrix = np.concatenate((weight_matrix, [bias]))

        for start_neuron, start_neuron_weights in enumerate(weight_matrix):
            if _input is not None:
                if _input[start_neuron] == 0:
                    continue

            max_end_neuron = np.argmax(np.abs(start_neuron_weights))
            weight = start_neuron_weights[max_end_neuron]

            start_pos = start_pos_dict[start_neuron]
            end_pos = end_pos_dict[max_end_neuron]

            Weights.draw_connection(visualizer, start_pos, end_pos, weight)

    @staticmethod
    def draw_weights(visualizer: "brain_visualizer.BrainVisualizer", start_pos_dict: dict, end_pos_dict: dict,
                     weight_matrix) -> None:

        start_neurons_drawn = np.zeros(len(start_pos_dict.keys()))
        end_neurons_drawn = np.zeros(len(end_pos_dict.keys()))

        for (start_neuron, end_neuron), weight in np.ndenumerate(weight_matrix):
            if weight != 0 and (
                    (weight > 0.0 and visualizer.positive_weights) or (weight < 0.0 and visualizer.negative_weights)):

                if visualizer.draw_threshold and abs(weight) < visualizer.draw_threshold:
                    start_drawn = bool(start_neurons_drawn[start_neuron])
                    end_drawn = bool(end_neurons_drawn[end_neuron])
                    if start_drawn and end_drawn:
                        continue

                    if not start_drawn:
                        start_neurons_drawn[start_neuron] = 1

                    if not end_drawn:
                        end_neurons_drawn[end_neuron] = 1

                start_pos = start_pos_dict[start_neuron]
                end_pos = end_pos_dict[end_neuron]

                Weights.draw_connection(visualizer, start_pos, end_pos, weight)
