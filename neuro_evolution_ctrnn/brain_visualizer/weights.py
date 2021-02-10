import math
from typing import Tuple
import logging
import pygame
import numpy as np

from brain_visualizer import brain_visualizer


class Weights:
    # Initialize weight mode flags
    WEIGHT_ALL = 1 << 0
    WEIGHT_MAX = 1 << 1
    IGNORE_ZERO_INPUT = 1 << 2

    @staticmethod
    def weight_flag_to_str(weight_flag: int):
        output_str = ""
        if weight_flag & Weights.WEIGHT_ALL:
            output_str += "All weights"
        elif weight_flag & Weights.WEIGHT_MAX:
            output_str += "Maximum weights"
        elif weight_flag & Weights.IGNORE_ZERO_INPUT:
            output_str += ", ignore inputs with zero value"
        else:
            raise RuntimeError("Weight flag '{}' not known.".format(weight_flag))

        return output_str

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
        if is_input:
            if weight > 0.0:
                weight_color = visualizer.color_input_connections_positive
            else:
                weight_color = visualizer.color_input_connections_negative
        else:
            if weight > 0.0:
                weight_color = visualizer.color_positive_weight
            else:
                weight_color = visualizer.color_negative_weight

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
    def check_zero_input(is_input: bool, draw_weight_mode: int, start_neuron: int, in_values: np.ndarray):
        if is_input and draw_weight_mode & Weights.IGNORE_ZERO_INPUT:
            if in_values is not None:
                if in_values[start_neuron] == 0:
                    return True
            else:
                raise RuntimeError("""Disabling weights which come from inputs with a value of zero requires to
                provide the input values to the draw weights function.""")
        return False

    @staticmethod
    def draw_maximum_weights(visualizer: "brain_visualizer.BrainVisualizer", start_pos_dict: dict, end_pos_dict: dict,
                             weight_matrix, is_input: bool = False, in_values: np.ndarray = None) -> None:
        if is_input and visualizer.rgb_input:
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
            Weights.check_zero_input(is_input, visualizer.draw_weight_mode, start_neuron, in_values)

            max_end_neuron = np.argmax(np.abs(start_neuron_weights))
            weight = start_neuron_weights[max_end_neuron]

            logging.debug(max_end_neuron)
            logging.debug(start_neuron_weights)
            logging.debug("----")
            logging.debug(end_pos_dict)

            start_pos = start_pos_dict[start_neuron]
            end_pos = end_pos_dict[max_end_neuron]

            Weights.draw_connection(visualizer, start_pos, end_pos, weight, is_input)

    @staticmethod
    def draw_weights(visualizer: "brain_visualizer.BrainVisualizer", start_pos_dict: dict, end_pos_dict: dict,
                     weight_matrix, is_input: bool = False, in_values: np.ndarray = None) -> None:
        weight_matrix = weight_matrix.T
        if visualizer.draw_weight_mode & Weights.WEIGHT_ALL:
            for (start_neuron, end_neuron), weight in np.ndenumerate(weight_matrix):
                Weights.check_zero_input(is_input, visualizer.draw_weight_mode, start_neuron, in_values)

                if (weight != 0 and
                        ((visualizer.positive_weights and weight > 0.0) or
                         (visualizer.negative_weights and weight < 0.0))):

                    if visualizer.draw_threshold and abs(weight) < visualizer.draw_threshold:
                        continue

                    start_pos = start_pos_dict[start_neuron]
                    end_pos = end_pos_dict[end_neuron]

                    Weights.draw_connection(visualizer, start_pos, end_pos, weight)
        # TODO wenn die Weights.WEIGHT_MAX usw. flags sind, warum ist dann hier elseif?
        elif visualizer.draw_weight_mode & Weights.WEIGHT_MAX:
            Weights.draw_maximum_weights(visualizer, start_pos_dict, end_pos_dict, weight_matrix, is_input, in_values)
        else:
            raise RuntimeError("""The specified drawing mode for the weights '{}' is not supported please choose another
             one.""".format(visualizer.draw_weight_mode))
