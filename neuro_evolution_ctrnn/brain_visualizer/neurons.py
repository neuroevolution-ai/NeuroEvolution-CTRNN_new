import logging
from typing import Tuple

import numpy as np
import pygame

from brain_visualizer.color import Colors
from brain_visualizer import brain_visualizer


class Neurons:

    @staticmethod
    def draw_neurons(visualizer: "brain_visualizer.BrainVisualizer", positions: dict, value_dict: np.ndarray,
                     color_clipping_range: int, negative_color: Tuple[int, int, int],
                     neutral_color: Tuple[int, int, int], positive_color: Tuple[int, int, int], radius: int,
                     matrix: bool = False, weight_neuron: bool = False) -> None:
        number_neurons_per_color = 0
        draw_text = visualizer.neuron_text

        if visualizer.rgb_input:
            draw_text = False
            number_neurons_per_color = visualizer.input_shape[0] * visualizer.input_shape[1]

        counter = 0
        for neuron in range(len(positions)):
            position = positions[neuron]
            pos_x = int(position[0])
            pos_y = int(position[1])

            if matrix:
                val = value_dict[neuron, neuron]
                color_val = val / color_clipping_range
            else:
                val = value_dict[neuron]
                color_val = val / color_clipping_range

            if weight_neuron:
                radius += int(abs(val))

            if visualizer.rgb_input:
                draw_text = False
                color_val = min(255, int(val * 256))
                if counter < number_neurons_per_color:
                    interpolated_color = (color_val, 0, 0)
                elif counter < 2 * number_neurons_per_color:
                    interpolated_color = (0, color_val, 0)
                else:
                    interpolated_color = (0, 0, color_val)

                counter += 1
                text_surface = None
            else:
                # Avoid program crash if clipping range is invalid
                if color_val > 1 or color_val < -1:
                    color_val = 1
                    Neurons.color_logging(visualizer, color_clipping_range)

                if color_val <= 0:
                    interpolated_color = Colors.interpolate_color(neutral_color, negative_color, abs(color_val))
                    text_surface = visualizer.my_font.render(("%.5s" % val), True, Colors.black)
                else:
                    interpolated_color = Colors.interpolate_color(neutral_color, positive_color, color_val)
                    text_surface = visualizer.my_font.render(("%.5s" % val), True, Colors.white)

            # Draw Circle and Text
            pygame.draw.circle(visualizer.screen, interpolated_color, (pos_x, pos_y), radius)

            if draw_text:
                visualizer.screen.blit(text_surface, ((pos_x - 16), (pos_y - 7)))

    @staticmethod
    def color_logging(visualizer: "brain_visualizer.BrainVisualizer", min_max: int) -> None:
        if min_max == visualizer.color_clipping_range[0]:
            var = "Input"
        elif min_max == visualizer.color_clipping_range[1]:
            var = "Graph"
        elif min_max == visualizer.color_clipping_range[2]:
            var = "Output"
        else:
            var = "Other"

        logging.warning("Please increase the clipping range for: " + var)
