import pygame
import numpy
import math


class Weights:

    @staticmethod
    def draw_weights(brain_visualizer, start_pos_dict, end_pos_dict, weight_matrix, positive_weights, negative_weights, direction):
        for (start_neuron, end_neuron), weight in numpy.ndenumerate(weight_matrix):
            if weight != 0:
                if (weight > 0.0 and positive_weights) or (weight < 0.0 and negative_weights):
                    start_pos = start_pos_dict[start_neuron]
                    end_pos = end_pos_dict[end_neuron]

                    if weight > 0.0:
                        weight_color = brain_visualizer.color_positive_weight
                    else:
                        weight_color = brain_visualizer.color_negative_weight

                    width = int(abs(weight)) + brain_visualizer.weight_val
                    if brain_visualizer.weight_val == 0 and width < 1:
                        width = 1

                    if direction:
                        # Winkel der Linien zwischen den beiden Punkten zur x-Achse
                        rotation = math.atan2((end_pos[1] - start_pos[1]), (end_pos[0] - start_pos[0]))
                        # Punkt, Winkel und Länge der Linie für Endpunkt des Pfeils
                        trirad = 5 + width
                        arrowLength = (-1 * (brain_visualizer.neuron_radius + trirad + 5))

                        arrowEnd = (
                            end_pos[0] + arrowLength * math.cos(rotation), end_pos[1] + arrowLength * math.sin(rotation))
                        if rotation != 0:
                            Weights.arrow(brain_visualizer.screen, weight_color, weight_color, start_pos, arrowEnd, trirad,
                                          width)
                    elif not direction:
                        pygame.draw.line(brain_visualizer.screen, weight_color, (start_pos[0], start_pos[1]), (endPosX, endPosY), width)

    @staticmethod
    def arrow(screen, color, tricolor, start, end, trirad, width):
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
